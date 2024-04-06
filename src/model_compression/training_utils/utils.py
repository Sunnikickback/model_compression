import dataclasses
import json
import csv
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

# +
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


# +
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
class TrainConfig:
    weight_decay: float = 0.01
    max_steps: int = -1
    gradient_accumulation_steps: int = 1
    num_train_epochs: int = 30
    warmup_ratio: float = 0.06
    learning_rate: float = 0.00001
    adam_epsilon:float = 1e-8
    max_grad_norm:float = 1.0
    train_batch_size: int = 16 
    eval_batch_size:int = 32
    eval_and_save_steps:float = 500

