from src.model_compression.training_utils.processors import (
                                           BoolqProcessor, WicProcessor, WscProcessor,
                                           # CopaProcessor, MultircProcessor, RteProcessor,
                                           # RecordProcessor,CbProcessor,
                                            # DiagnosticBroadProcessor, DiagnosticGenderProcessor,
                                        )


processors = {
    # "ax-b": DiagnosticBroadProcessor,
    # "ax-g": DiagnosticGenderProcessor,
    "boolq": BoolqProcessor,
    # "cb": CbProcessor,
    # "copa": CopaProcessor,
    # "multirc": MultircProcessor,
    # "record": RecordProcessor,
    # "rte": RteProcessor,
    "wic": WicProcessor,
    "wsc": WscProcessor,
}

output_modes = {
    # "ax-b": "classification",
    # "ax-g": "classification",
    "boolq": "classification",
    # "cb": "classification",
    # "copa": "classification",
    # "multirc": "classification",
    # "record": "classification",
    # "rte": "classification",
    "wic": "span_classification",
    "wsc": "span_classification",
}

tasks_metrics = {
    "boolq": "acc",
    # "cb": "acc_and_f1",
    # "copa": "acc",
    # "multirc": "em_and_f1",
    # "record": "em_and_f1",
    # "rte": "acc",
    "wic": "acc",
    "wsc": "acc_and_f1",
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

def superglue_convert_examples_to_features(
    examples,
    tokenizer,
    max_length=512,
    task=None,
    label_list=None,
    output_mode=None,
    pad_on_left=False,
    pad_token=0,
    pad_token_segment_id=0,
    mask_padding_with_zero=True,
):

    if task is not None:
        processor = superglue_processors[task]()
        if label_list is None:
            label_list = processor.get_labels()
        if output_mode is None:
            output_mode = superglue_output_modes[task]

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

                # check that the length of the input ids is expected (not necessarily the exact ids)
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
        if pad_on_left:
            assert False, "Not implemented correctly wrt span tracking!"
            input_ids = ([pad_token] * padding_length) + input_ids
            attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
            token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids

        else:
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

def load_and_cache_examples(task, tokenizer, data_dir, split="train"):
    processor = processors[task]()
    output_mode = output_modes[task]
    label_list = processor.get_labels()

    if split == "train":
        get_examples = processor.get_train_examples
    elif split == "dev":
        get_examples = processor.get_dev_examples
    elif split == "test":
        get_examples = processor.get_test_examples

    examples = get_examples(data_dir)

    features = superglue_convert_examples_to_features(
        examples,
        tokenizer,
        label_list=label_list,
        max_length=args.max_seq_length,
        output_mode=output_mode,
        pad_on_left=0,
        pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
        pad_token_segment_id=0,
    )

    # Convert to Tensors and build dataset
    all_guids = torch.tensor([f.guid for f in features], dtype=torch.long)
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_seq_lengths = torch.tensor([f.seq_length for f in features], dtype=torch.long)
    if output_mode in ["classification", "span_classification"]:
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

    if output_mode in ["span_classification"]:
        # all_starts = torch.tensor([[s[0] for s in f.span_locs] for f in features], dtype=torch.long)
        # all_ends = torch.tensor([[s[1] for s in f.span_locs] for f in features], dtype=torch.long)
        all_spans = torch.tensor([f.span_locs for f in features])
        dataset = TensorDataset(
            all_input_ids, all_attention_mask, all_token_type_ids, all_labels, all_spans, all_seq_lengths, all_guids
        )
    else:
        dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels,
                                all_seq_lengths, all_guids)

    if args.task_name == "record" and split in ["dev", "test"]:
        answers = processor.get_answers(args.data_dir, split)
        return dataset, answers
    else:
        return dataset