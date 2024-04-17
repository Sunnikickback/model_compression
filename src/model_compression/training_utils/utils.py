import argparse
import dataclasses
import json
import csv
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import numpy as np
import torch
from transformers import RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification
from model_compression.training_utils.modules import RobertaForSpanClassification

MODEL_CLASSES = {
    "roberta": (
        RobertaConfig,
        RobertaTokenizer,
        {"classification": RobertaForSequenceClassification, "span_classification": RobertaForSpanClassification},
    ),
}

tasks_num_spans = {
    "wic": 2,
    "wsc": 2,
}

task_metrics = {
    "boolq": "acc",
    "cb": "acc_and_f1",
    "copa": "acc",
    "multirc": "em_and_f1",
    "rte": "acc",
    "wic": "acc",
    "wsc": "acc_and_f1",
}

output_modes = {
    "boolq": "classification",
    "cb": "classification",
    "copa": "classification",
    "multirc": "classification",
    "rte": "classification",
    "wic": "span_classification",
    "wsc": "span_classification",
}
tasks_num_labels = {
    "boolq": 2,
    "cb": 3,
    "copa": 2,
    "rte": 2,
    "wic": 2,
    "wsc": 2,
}


@dataclass
class InputExample:
    guid: str
    text_a: str
    text_b: Optional[str] = None
    label: Optional[str] = None
    seq_length: int = None

    def to_json_string(self):
        return json.dumps(dataclasses.asdict(self), indent=2) + "\n"


class DataProcessor:
    def get_example_from_tensor_dict(self, tensor_dict):
        raise NotImplementedError()

    def get_train_examples(self, data_dir):
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        raise NotImplementedError()

    def get_labels(self):
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        with open(input_file, "r", encoding="utf-8-sig") as f:
            return list(csv.reader(f, delimiter="\t", quotechar=quotechar))

    @classmethod
    def _read_jsonl(cls, input_file, quotechar=None):
        with open(input_file, "r", encoding="utf-8-sig") as f:
            return [json.loads(l) for l in f]


@dataclass(frozen=True)
class SpanClassificationExample(object):
    guid: str
    text_a: str
    spans_a: List[Tuple[int]]
    text_b: Optional[str] = None
    spans_b: Optional[List[Tuple[int]]] = None
    label: Optional[str] = None
    seq_length: int = None

    def to_json_string(self):
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


@dataclass(frozen=True)
class InputFeatures:
    guid: List[int]
    input_ids: List[int]
    attention_mask: Optional[List[int]] = None
    token_type_ids: Optional[List[int]] = None
    label: Optional[Union[int, float]] = None
    seq_length: int = None

    def to_json_string(self):
        return json.dumps(dataclasses.asdict(self)) + "\n"


@dataclass(frozen=True)
class SpanClassificationFeatures(object):
    guid: List[int]
    input_ids: List[int]
    span_locs: List[Tuple[int]]
    attention_mask: Optional[List[int]] = None
    token_type_ids: Optional[List[int]] = None
    label: Optional[Union[int, float]] = None
    seq_length: int = None

    def to_json_string(self):
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


@dataclass
class PruneConfig:
    dont_normalize_importance_by_layer: True
    target_ffn_dim: int
    target_num_heads: int
    eval_batch_size: int = 32


@dataclass
class DistilConfig:
    state_loss_ratio: float
    state_distill_cs: bool
    att_loss_ratio: float

    @staticmethod
    def from_args(args) -> 'DistilConfig':
        return DistilConfig(
            state_loss_ratio=args.state_loss_ratio,
            state_distill_cs=args.state_distill_cs,
            att_loss_ratio=args.att_loss_ratio
        )


@dataclass
class TrainConfig:
    weight_decay: float = 0.01
    max_steps: int = -1
    gradient_accumulation_steps: int = 1
    num_train_epochs: int = 30
    warmup_ratio: float = 0.06
    learning_rate: float = 1e-5
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    train_batch_size: int = 32
    eval_batch_size: int = 32
    eval_and_save_steps: float = 500

    @staticmethod
    def from_args(args) -> 'TrainConfig':
        return TrainConfig(
            weight_decay=args.weight_decay,
            max_steps=args.max_steps,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            num_train_epochs=args.num_train_epochs,
            learning_rate=args.learning_rate,
            warmup_ratio=args.warmup_ratio,
            adam_epsilon=args.adam_epsilon,
            max_grad_norm=args.max_grad_norm,
            train_batch_size=args.train_batch_size,
            eval_batch_size=args.eval_batch_size,
            eval_and_save_steps=args.eval_and_save_steps
        )


@dataclass
class ModelConfig:
    task_name: str
    model_type: str
    model_checkpoint: str
    data_dir: str


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args(mode):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=None, type=str)
    parser.add_argument("--model_type", default=None, type=str, required=True)
    parser.add_argument("--model_checkpoint", default=None, type=str, required=True)
    parser.add_argument("--task_name", default=None, type=str, required=True)
    parser.add_argument("--do_lower_case", action="store_true")
    if mode == "distillation":
        parser.add_argument("--teacher_path", default=None, type=str, required=True)
        parser.add_argument("--state_distill_cs", action="store_true")
        parser.add_argument("--att_loss_ratio", default=0.0, type=float)
        parser.add_argument("--state_loss_ratio", default=0.1, type=float)
    if mode == "pruning":
        parser.add_argument("--target_ffn_dim", default=3072, type=int)
        parser.add_argument("--target_num_heads", default=12, type=int)
        parser.add_argument("--dont_normalize_importance_by_layer", action="store_true")
    if mode == "quantization":
        parser.add_argument("--nodes_to_exclude", default=None, type=str)
    if mode == "training" or mode == "distillation":
        parser.add_argument("--weight_decay", default=0.01, type=float)
        parser.add_argument("--warmup_ratio", default=0.06, type=float)
        parser.add_argument("--max_steps", default=-1, type=int)
        parser.add_argument("--gradient_accumulation_steps", default=1, type=int)
        parser.add_argument("--num_train_epochs", default=30, type=int)
        parser.add_argument("--learning_rate", default=0.00001, type=float)
        parser.add_argument("--adam_epsilon", default=1e-8, type=float)
        parser.add_argument("--max_grad_norm", default=1.0, type=float)
        parser.add_argument("--train_batch_size", default=16, type=int)
        parser.add_argument("--eval_batch_size", default=32, type=int)
        parser.add_argument("--eval_and_save_steps", default=500, type=int)
        parser.add_argument("--max_seq_length", default=512, type=int)

    args = parser.parse_args()
    return args


def get_model(args):
    from model_compression.training_utils.processors import superglue_processors as processors
    args.task_name = args.task_name.lower()
    assert args.task_name in processors, f"Task {args.task_name} not found!"
    args.model_type = args.model_type.lower()
    output_mode = output_modes[args.task_name]
    config_class, tokenizer_class, model_classes = MODEL_CLASSES[args.model_type]
    model_class = model_classes[output_mode]
    processor = processors[args.task_name]()
    label_list = processor.get_labels()
    num_labels = len(label_list)
    config = config_class.from_pretrained(
        args.model_checkpoint,
        num_labels=num_labels,
        finetuning_task=args.task_name,
    )
    if output_modes[args.task_name] == "span_classification":
        config.num_spans = tasks_num_spans[args.task_name]
    tokenizer = tokenizer_class.from_pretrained(args.model_checkpoint, do_lower_case=args.do_lower_case)
    model = model_class.from_pretrained(args.model_checkpoint, config=config)
    model_config = ModelConfig(data_dir=args.data_dir, model_type=args.model_type,
                               model_checkpoint=args.model_checkpoint, task_name=args.task_name)
    return model, tokenizer, model_config

