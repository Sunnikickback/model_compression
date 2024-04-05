import dataclasses
import os
import json
from collections import defaultdict
import operator
import csv
from typing import Optional
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

# +
task_metrics = {
    "boolq": "acc",
    # "cb": "acc_and_f1",
    # "copa": "acc",
    # "multirc": "em_and_f1",
    # "record": "em_and_f1",
    # "rte": "acc",
    "wic": "acc",
    "wsc": "acc_and_f1",
}

output_modes = {
    # "ax-b": "classification",
    # "ax-g": "classification",
    "boolq": "classification",
    # "cb": "classification",
    # "copa": "classification",
    # "multirc": "classification",
    # "rte": "classification",
    "wic": "span_classification",
    "wsc": "span_classification",
}
tasks_num_labels = {
    # "ax-b": 2,
    # "ax-g": 2,
    "boolq": 2,
    # "cb": 3,
    # "copa": 2,
    # "rte": 2,
    "wic": 2,
    "wsc": 2,
}


# -

@dataclass
class InputExample:
    guid: str
    text_a: str
    text_b: Optional[str] = None
    label: Optional[str] = None
    seq_length: int = None

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
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
    """
    A single training/test example for simple span classification.

    Args:
        guid: Unique id for the example.
        text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
        spans_a: list. List of tuples of ints corresponding to the character locations in text_a
            of the spans of interest.
        text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
        spans_b: list. List of tuples of ints corresponding to the character locations in text_b
        label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
    """

    guid: str
    text_a: str
    spans_a: List[Tuple[int]]
    text_b: Optional[str] = None
    spans_b: Optional[List[Tuple[int]]] = None
    label: Optional[str] = None
    seq_length: int = None

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


@dataclass(frozen=True)
class InputFeatures:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.

    Args:
        guid: Example ID, as a list of ints
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
        token_type_ids: (Optional) Segment token indices to indicate first and second
            portions of the inputs. Only some models use them.
        label: (Optional) Label corresponding to the input. Int for classification problems,
            float for regression problems.
    """

    guid: List[int]
    input_ids: List[int]
    attention_mask: Optional[List[int]] = None
    token_type_ids: Optional[List[int]] = None
    label: Optional[Union[int, float]] = None
    seq_length: int = None

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self)) + "\n"


# +
@dataclass(frozen=True)
class SpanClassificationFeatures(object):
    """
    A single set of features of data.

    Args:
        guid: Example ID, as a list of ints
        input_ids: Indices of input sequence tokens in the vocabulary.
        span_locs: List of spans, length 2 lists of indices.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
        token_type_ids: Segment token indices to indicate first and second portions of the inputs.
        label: Label corresponding to the input
    """

    guid: List[int]
    input_ids: List[int]
    span_locs: List[Tuple[int]]
    attention_mask: Optional[List[int]] = None
    token_type_ids: Optional[List[int]] = None
    label: Optional[Union[int, float]] = None
    seq_length: int = None

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
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

