import torch
from torch.utils.data import TensorDataset

from model_compression.training_utils.processors import superglue_processors as processors
from model_compression.training_utils.utils import SpanClassificationExample, InputFeatures
from model_compression.training_utils.utils import output_modes, SpanClassificationFeatures


def tokenize_tracking_span(tokenizer, text, spans):
    toks = tokenizer.encode_plus(text, return_token_type_ids=True)
    full_toks = toks["input_ids"]
    prefix_len = len(tokenizer.decode(full_toks[:1])) + 1  # add a space
    len_covers = []
    for i in range(2, len(full_toks)):
        partial_txt_len = len(tokenizer.decode(full_toks[:i], clean_up_tokenization_spaces=False))
        len_covers.append(partial_txt_len - prefix_len)

    span_locs = []
    for start, end in spans:
        start_tok, end_tok = None, None
        for tok_n, len_cover in enumerate(len_covers):
            if len_cover >= start and start_tok is None:
                start_tok = tok_n + 1
            if len_cover >= end:
                assert start_tok is not None
                end_tok = tok_n + 1
                break
        assert start_tok is not None, "start_tok is None!"
        assert end_tok is not None, "end_tok is None!"
        span_locs.append((start_tok, end_tok))
    return toks, span_locs


def superglue_convert_examples_to_features(
        examples,
        tokenizer,
        max_length=512,
        task=None,
        label_list=None,
        output_mode=None,
        pad_token=0,
        pad_token_segment_id=0,
        mask_padding_with_zero=True,
):
    if task is not None:
        processor = processors[task]()
        if label_list is None:
            label_list = processor.get_labels()
        if output_mode is None:
            output_mode = output_modes[task]

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if isinstance(example, SpanClassificationExample):
            inputs_a, span_locs_a = tokenize_tracking_span(tokenizer, example.text_a, example.spans_a)
            if example.spans_b is not None:
                inputs_b, span_locs_b = tokenize_tracking_span(tokenizer, example.text_b, example.spans_b)
                num_non_special_tokens = len(inputs_a["input_ids"]) + len(inputs_b["input_ids"]) - 4

                inputs = tokenizer.encode_plus(
                    example.text_a,
                    example.text_b,
                    add_special_tokens=True,
                    max_length=max_length,
                    return_token_type_ids=True,
                    truncation=True,
                )
                num_joiner_specials = len(inputs["input_ids"]) - num_non_special_tokens - 2
                offset = len(inputs_a["input_ids"]) - 1 + num_joiner_specials - 1
                span_locs_b = [(s + offset, e + offset) for s, e in span_locs_b]
                span_locs = span_locs_a + span_locs_b
                input_ids = inputs["input_ids"]
                token_type_ids = inputs["token_type_ids"]

                if num_joiner_specials == 1:
                    tmp = inputs_a["input_ids"] + inputs_b["input_ids"][1:]
                elif num_joiner_specials == 2:
                    tmp = inputs_a["input_ids"] + inputs_b["input_ids"]
                else:
                    assert False, "Something is wrong"

                assert len(input_ids) == len(tmp), "Span tracking tokenization produced inconsistent result!"

            else:
                input_ids, token_type_ids = inputs_a["input_ids"], inputs_a["token_type_ids"]
                span_locs = span_locs_a
        else:
            inputs = tokenizer.encode_plus(
                example.text_a,
                example.text_b,
                add_special_tokens=True,
                max_length=max_length,
                return_token_type_ids=True,
                truncation=True,
            )
            input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        seq_length = len(input_ids)
        padding_length = max_length - len(input_ids)

        input_ids = input_ids + ([pad_token] * padding_length)
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
        assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(
            len(attention_mask), max_length
        )
        assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(
            len(token_type_ids), max_length
        )
        if output_mode in ["classification", "span_classification"]:
            label = label_map[example.label]
        elif output_mode == "regression":
            label = float(example.label)
        else:
            raise KeyError(output_mode)

        if isinstance(example, SpanClassificationExample):
            feats = SpanClassificationFeatures(
                guid=example.guid,
                input_ids=input_ids,
                span_locs=span_locs,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                label=label,
                seq_length=seq_length,
            )
        else:
            feats = InputFeatures(
                guid=example.guid,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                label=label,
                seq_length=seq_length,
            )

        features.append(feats)

    return features


def load_and_cache_examples(tokenizer, model_config, split="train", max_seq_length=512):
    processor = processors[model_config.task_name]()
    output_mode = output_modes[model_config.task_name]
    label_list = processor.get_labels()

    if split == "train":
        get_examples = processor.get_train_examples
    elif split == "dev":
        get_examples = processor.get_dev_examples
    else:
        get_examples = processor.get_test_examples

    examples = get_examples(model_config.data_dir)

    features = superglue_convert_examples_to_features(
        examples,
        tokenizer,
        label_list=label_list,
        max_length=max_seq_length,
        output_mode=output_mode,
        pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
        pad_token_segment_id=0,
    )

    all_guids = torch.tensor([f.guid for f in features], dtype=torch.long)
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_seq_lengths = torch.tensor([f.seq_length for f in features], dtype=torch.long)
    if output_mode in ["classification", "span_classification", "ner_classification"]:
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

    if output_mode in ["span_classification"]:
        all_spans = torch.tensor([f.span_locs for f in features])
        dataset = TensorDataset(
            all_input_ids, all_attention_mask, all_token_type_ids,
            all_labels, all_spans, all_seq_lengths, all_guids
        )
    else:
        dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels,
                                all_seq_lengths, all_guids)

    return dataset
