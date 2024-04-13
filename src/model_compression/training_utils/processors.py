import os
import json
from collections import defaultdict
import operator
from .utils import DataProcessor, InputExample, SpanClassificationExample


class BoolqProcessor(DataProcessor):
    def get_example_from_tensor_dict(self, tensor_dict):
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["question"].numpy().decode("utf-8"),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        return self._create_examples(self._read_jsonl(os.path.join(data_dir, "train.jsonl")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(self._read_jsonl(os.path.join(data_dir, "val.jsonl")), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(self._read_jsonl(os.path.join(data_dir, "test.jsonl")), "test")

    def get_labels(self):
        return [True, False]

    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = line["idx"]
            text_a = line["passage"]
            text_b = line["question"]
            label = line["label"] if "label" in line else False
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def write_preds(self, preds, ex_ids, out_dir):
        preds_with_exids = list(zip(preds, ex_ids))
        preds_with_exids.sort(key = operator.itemgetter(1))
        idx2label = {i: label for i, label in enumerate(self.get_labels())}
        with open(os.path.join(out_dir, "BoolQ.jsonl"), "w") as pred_fh:
            for idx, pred_exid in enumerate(preds_with_exids):
                pred_label = idx2label[int(pred_exid[0])]
                pred_fh.write(f"{json.dumps({'idx': idx, 'label': 'true' if pred_label else 'false'})}\n")


class WscProcessor(DataProcessor):
    def get_example_from_tensor_dict(self, tensor_dict):
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence1"].numpy().decode("utf-8"),
            tensor_dict["sentence2"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        return self._create_examples(self._read_jsonl(os.path.join(data_dir, "train.jsonl")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(self._read_jsonl(os.path.join(data_dir, "val.jsonl")), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(self._read_jsonl(os.path.join(data_dir, "test.jsonl")), "test")

    def get_labels(self):
        return [True, False]

    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = line["idx"]
            text_a = line["text"]
            span_start1 = line["target"]["span1_index"]
            span_start2 = line["target"]["span2_index"]
            span_end1 = span_start1 + len(line["target"]["span1_text"])
            span_end2 = span_start2 + len(line["target"]["span2_text"])
            span1 = (span_start1, span_end1)
            span2 = (span_start2, span_end2)
            label = line["label"] if "label" in line else False
            examples.append(SpanClassificationExample(guid=guid, text_a=text_a, spans_a=[span1, span2], label=label))
        return examples

    def write_preds(self, preds, ex_ids, out_dir):
        preds = preds[ex_ids]
        idx2label = {i: label for i, label in enumerate(self.get_labels())}
        with open(os.path.join(out_dir, "WSC.jsonl"), "w") as pred_fh:
            for idx, pred in enumerate(preds):
                pred_label = idx2label[int(pred)]
                pred_fh.write(f"{json.dumps({'idx': idx, 'label': pred_label})}\n")

class WicProcessor(DataProcessor):
    def get_example_from_tensor_dict(self, tensor_dict):
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence1"].numpy().decode("utf-8"),
            tensor_dict["sentence2"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        return self._create_examples(self._read_jsonl(os.path.join(data_dir, "train.jsonl")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(self._read_jsonl(os.path.join(data_dir, "val.jsonl")), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(self._read_jsonl(os.path.join(data_dir, "test.jsonl")), "test")

    def get_labels(self):
        return [True, False]

    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = line["idx"]
            text_a = line["sentence1"]
            text_b = line["sentence2"]
            span_a = (line["start1"], line["end1"])
            span_b = (line["start2"], line["end2"])
            label = line["label"] if "label" in line else False
            examples.append(
                SpanClassificationExample(
                    guid=guid, text_a=text_a, spans_a=[span_a], text_b=text_b, spans_b=[span_b], label=label
                )
            )
        return examples

    def write_preds(self, preds, ex_ids, out_dir):
        preds_with_exids = list(zip(preds, ex_ids))
        preds_with_exids.sort(key = operator.itemgetter(1))
        idx2label = {i: label for i, label in enumerate(self.get_labels())}
        with open(os.path.join(out_dir, "WiC.jsonl"), "w") as pred_fh:
            for idx, pred_exid in enumerate(preds_with_exids):
                pred_label = idx2label[int(pred_exid[0])]
                pred_fh.write(f"{json.dumps({'idx': idx, 'label': 'true' if pred_label else 'false'})}\n")

class CbProcessor(DataProcessor):
    def get_example_from_tensor_dict(self, tensor_dict):
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["question"].numpy().decode("utf-8"),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        return self._create_examples(self._read_jsonl(os.path.join(data_dir, "train.jsonl")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(self._read_jsonl(os.path.join(data_dir, "val.jsonl")), "dev_matched")

    def get_test_examples(self, data_dir):
        return self._create_examples(self._read_jsonl(os.path.join(data_dir, "test.jsonl")), "test")

    def get_labels(self):
        return ["entailment", "contradiction", "neutral"]

    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = line["idx"]
            text_a = line["premise"]
            text_b = line["hypothesis"]
            label = line["label"] if "label" in line else "contradiction"
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def write_preds(self, preds, ex_ids, out_dir):
        preds_with_exids = list(zip(preds, ex_ids))
        preds_with_exids.sort(key = operator.itemgetter(1))
        idx2label = {i: label for i, label in enumerate(self.get_labels())}
        with open(os.path.join(out_dir, "CB.jsonl"), "w") as pred_fh:
            for idx, pred_exid in enumerate(preds_with_exids):
                pred_label = idx2label[int(pred_exid[0])]
                pred_fh.write(f"{json.dumps({'idx': idx, 'label': pred_label})}\n")
class CopaProcessor(DataProcessor):
    def get_example_from_tensor_dict(self, tensor_dict):
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["question"].numpy().decode("utf-8"),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        return self._create_examples(self._read_jsonl(os.path.join(data_dir, "train.jsonl")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(self._read_jsonl(os.path.join(data_dir, "val.jsonl")), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(self._read_jsonl(os.path.join(data_dir, "test.jsonl")), "test")

    def get_labels(self):
        return [0, 1]

    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = line["idx"]
            label = line["label"] if "label" in line else 0
            premise = line["premise"][:-1]
            choice1 = line["choice1"]
            choice2 = line["choice2"]
            joiner = "because" if line["question"] == "cause" else "so"
            text_a = f"{premise} {joiner} {choice1}"
            text_b = f"{premise} {joiner} {choice2}"
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def write_preds(self, preds, ex_ids, out_dir):
        preds_with_exids = list(zip(preds, ex_ids))
        preds_with_exids.sort(key = operator.itemgetter(1))
        idx2label = {i: label for i, label in enumerate(self.get_labels())}
        with open(os.path.join(out_dir, "COPA.jsonl"), "w") as pred_fh:
            for idx, pred_exid in enumerate(preds_with_exids):
                pred_label = idx2label[int(pred_exid[0])]
                pred_fh.write(f"{json.dumps({'idx': idx, 'label': pred_label})}\n")


class MultircProcessor(DataProcessor):
    def get_example_from_tensor_dict(self, tensor_dict):
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["question"].numpy().decode("utf-8"),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        return self._create_examples(self._read_jsonl(os.path.join(data_dir, "train.jsonl")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(self._read_jsonl(os.path.join(data_dir, "val.jsonl")), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(self._read_jsonl(os.path.join(data_dir, "test.jsonl")), "test")

    def get_labels(self):
        return [0, 1]

    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            passage_id = line["idx"]
            passage = line["passage"]["text"]
            for question_dict in line["passage"]["questions"]:
                question_id = question_dict["idx"]
                question = question_dict["question"]
                passage_and_question = " ".join([passage, question])
                for answer_dict in question_dict["answers"]:
                    answer_id = answer_dict["idx"]
                    guid = [passage_id, question_id, answer_id]
                    answer = answer_dict["text"]
                    label = answer_dict["label"] if "label" in answer_dict else 0
                    assert passage_and_question, "Empty passage and question!"
                    if answer == "":
                        if set_type == "train":
                            continue
                        else:
                            answer = "no"
                    examples.append(InputExample(guid=guid, text_a=passage_and_question, text_b=answer, label=label))
        return examples

    def write_preds(self, preds, ex_ids, out_dir):
        psg2qst2ans = defaultdict(lambda: defaultdict(dict))
        for pred, ex_id in zip(preds, ex_ids):
            psg_id, qst_id, ans_id = map(int, ex_id)
            psg2qst2ans[psg_id][qst_id][ans_id] = pred

        idx2label = {i: label for i, label in enumerate(self.get_labels())}
        with open(os.path.join(out_dir, "MultiRC.jsonl"), "w") as pred_fh:
            for psg_id, qst2ans in psg2qst2ans.items():
                psgs = []
                for qst_id, ans2pred in qst2ans.items():
                    anss = []
                    for ans_id, pred in ans2pred.items():
                        pred_label = idx2label[pred]
                        anss.append({"idx": ans_id, "label": pred_label})
                    psgs.append({"idx": qst_id, "answers": anss})
                pred_fh.write(f"{json.dumps({'idx': psg_id, 'passage': {'questions': psgs}})}\n")


class RteProcessor(DataProcessor):
    def get_example_from_tensor_dict(self, tensor_dict):
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence1"].numpy().decode("utf-8"),
            tensor_dict["sentence2"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        return self._create_examples(self._read_jsonl(os.path.join(data_dir, "train.jsonl")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(self._read_jsonl(os.path.join(data_dir, "val.jsonl")), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(self._read_jsonl(os.path.join(data_dir, "test.jsonl")), "test")

    def get_labels(self):
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = line["idx"]
            text_a = line["premise"]
            text_b = line["hypothesis"]
            label = line["label"] if "label" in line else "not_entailment"
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def write_preds(self, preds, ex_ids, out_dir):
        preds_with_exids = list(zip(preds, ex_ids))
        preds_with_exids.sort(key = operator.itemgetter(1))
        idx2label = {i: label for i, label in enumerate(self.get_labels())}
        with open(os.path.join(out_dir, "RTE.jsonl"), "w") as pred_fh:
            for idx, pred_exid in enumerate(preds_with_exids):
                pred_label = idx2label[int(pred_exid[0])]
                pred_fh.write(f"{json.dumps({'idx': idx, 'label': pred_label})}\n")

processors = {
    "boolq": BoolqProcessor,
    "cb": CbProcessor,
    "copa": CopaProcessor,
    "multirc": MultircProcessor,
    "rte": RteProcessor,
    "wic": WicProcessor,
    "wsc": WscProcessor,
}