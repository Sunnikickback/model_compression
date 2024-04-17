import os

import numpy as np
import torch
from onnxruntime import InferenceSession, SessionOptions
from onnxruntime.capi.onnxruntime_pybind11_state import RuntimeException
from onnxruntime.transformers.onnx_model_bert import BertOptimizationOptions
from onnxruntime.transformers.optimizer import optimize_model
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup

from model_compression.training_utils.processors import superglue_processors as processors
from model_compression.training_utils.datasets import load_and_cache_examples
from model_compression.training_utils.metrics import superglue_compute_metrics
from model_compression.training_utils.utils import TrainConfig, task_metrics, set_seed, get_model, parse_args, ModelConfig, output_modes


def evaluate(model, tokenizer, model_config, eval_batch_size, device,
             use_fixed_seq_length, split="dev", prefix="", use_tqdm=True, max_seq_length=512):
    output_mode = output_modes[model_config.task_name]
    results = {}

    eval_dataset = load_and_cache_examples(model_config=model_config, tokenizer=tokenizer, split=split,
                                           max_seq_length=max_seq_length)

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=eval_batch_size)

    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    ex_ids = None
    eval_dataloader = tqdm(eval_dataloader, desc="Evaluating") if use_tqdm else eval_dataloader
    for batch in eval_dataloader:
        model.eval()
        batch = tuple(t.to(device) for t in batch)
        guids = batch[-1]

        max_seq_length = batch[0].size(1)
        if use_fixed_seq_length:
            batch_seq_length = max_seq_length
        else:
            batch_seq_length = torch.max(batch[-2], 0)[0].item()

        if batch_seq_length < max_seq_length:
            inputs = {"input_ids": batch[0][:, :batch_seq_length].contiguous(),
                      "attention_mask": batch[1][:, :batch_seq_length].contiguous(),
                      "labels": batch[3]}
            if output_mode == "span_classification":
                inputs["spans"] = batch[4]
            inputs["token_type_ids"] = None
        else:
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
            if output_mode == "span_classification":
                inputs["spans"] = batch[4]
            inputs["token_type_ids"] = None

        with torch.no_grad():
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs["labels"].detach().cpu().numpy()
            ex_ids = [guids.detach().cpu().numpy()]
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)
            ex_ids.append(guids.detach().cpu().numpy())

    ex_ids = np.concatenate(ex_ids, axis=0)
    if output_mode in ["classification", "span_classification"] and model_config.task_name not in ["record"]:
        preds = np.argmax(preds, axis=1)
    elif output_mode == "regression":
        preds = np.squeeze(preds)
    if split != "test":
        output_dir = ""
        result = superglue_compute_metrics(model_config.task_name, preds, out_label_ids, ex_ids)
        results.update(result)
        output_eval_file = os.path.join(output_dir, prefix, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            for key in sorted(result.keys()):
                writer.write("%s = %s\n" % (key, str(result[key])))
    return results, preds, ex_ids


def train(train_dataset, model, tokenizer, model_config, train_config: TrainConfig, device="cuda:0"):
    model.to(device)
    output_mode = output_modes[model_config.task_name]

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=train_config.train_batch_size)

    if train_config.max_steps > 0:
        t_total = train_config.max_steps
        train_config.num_train_epochs = train_config.max_steps // (len(train_dataloader)
                                                                   // train_config.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // train_config.gradient_accumulation_steps * train_config.num_train_epochs
    num_warmup_steps = int(train_config.warmup_ratio * t_total)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": train_config.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=train_config.learning_rate, eps=train_config.adam_epsilon, )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=t_total
    )

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = range(epochs_trained, int(train_config.num_train_epochs))

    set_seed(1337)
    for epoch_n in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc=f"Epoch {epoch_n}")
        for step, batch in enumerate(epoch_iterator):

            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(device) for t in batch)
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
            if output_mode == "span_classification":
                inputs["spans"] = batch[4]
            inputs["token_type_ids"] = None

            outputs = model(**inputs)
            loss = outputs[0]

            if train_config.gradient_accumulation_steps > 1:
                loss = loss / train_config.gradient_accumulation_steps
            loss.backward()

            tr_loss += loss.item()
            epoch_iterator.set_description(f"Epoch {epoch_n} loss: {loss:.3f}")
            if (step + 1) % train_config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.max_grad_norm)

                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1

                if train_config.eval_and_save_steps > 0 and global_step % train_config.eval_and_save_steps == 0:
                    results, _, _ = evaluate(model, tokenizer, model_config=model_config,
                                             eval_batch_size=train_config.eval_batch_size,
                                             device=device, use_fixed_seq_length=False, use_tqdm=True)
                    print(results[task_metrics[model_config.task_name]])

            if 0 < train_config.max_steps <= global_step:
                epoch_iterator.close()
                break
        if 0 < train_config.max_steps <= global_step:
            break

    return global_step, tr_loss / global_step


def convert_model_to_onnx(model, tokenizer, model_config: ModelConfig):
    tokens = tokenizer.encode_plus("This is a sample input.")
    print(tokens)

    model.to(torch.device("cpu"))
    model.eval()

    print(">>>>>>>>> Model loaded.")

    input_names = ['input_ids', 'attention_mask']
    output_names = ['output_0']
    dynamic_axes = {
        'attention_mask': {
            0: 'batch',
            1: 'sequence'
        },
        'input_ids': {
            0: 'batch',
            1: 'sequence'
        },
        'output_0': {
            0: 'batch',
            1: 'sequence'
        }
    }

    model_args = (torch.tensor(tokens['input_ids']).unsqueeze(0),
                  torch.tensor(tokens['attention_mask']).unsqueeze(0))

    print(">>>>>>> ONNX conversion started!")
    torch.onnx.export(
        model,
        model_args,
        f=f"{model_config.model_type}_{model_config.task_name}.onnx",
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
        use_external_data_format=False,
        enable_onnx_checker=True,
        opset_version=11,
    )
    optimization_options = BertOptimizationOptions('bert')
    optimization_options.enable_gelu = True
    optimization_options.enable_layer_norm = True
    optimization_options.enable_attention = True
    optimization_options.enable_skip_layer_norm = True
    optimization_options.enable_embed_layer_norm = True
    optimization_options.enable_bias_skip_layer_norm = True
    optimization_options.enable_bias_gelu = True
    optimization_options.enable_gelu_approximation = False

    optimizer = optimize_model(f" {model_config.model_type}_{model_config.task_name}.onnx",
                               model_type='bert',
                               num_heads=0,
                               hidden_size=0,
                               optimization_options=optimization_options,
                               opt_level=0,
                               use_gpu=False,
                               only_onnxruntime=False)
    optimizer.save_model_to_file(f" {model_config.model_type}_{model_config.task_name}.onnx")

    print(f">>>>>>> Model converted into ONNX format and saved as:"
          f" {model_config.model_type}_{model_config.task_name}.onnx")

    try:
        onnx_options = SessionOptions()
        sess = InferenceSession(f" {model_config.model_type}_{model_config.task_name}.onnx", onnx_options)
        print("Model loaded successfully.")

        output_onnx = sess.run(None, {'input_ids': [tokens['input_ids']],
                                      'attention_mask': [tokens['attention_mask']]})
        print(output_onnx)
    except RuntimeException as re:
        print("Error while loading the model: {}".format(re))


def main():
    args = parse_args("training")
    model, tokenizer, model_config = get_model(args)
    train_dataset = load_and_cache_examples(tokenizer, model_config, max_seq_length=args.max_seq_length)
    train_config = TrainConfig().from_args(args)
    _, _ = train(train_dataset, model, tokenizer, model_config=model_config, train_config=train_config)
    result, preds, ex_ids = evaluate(model, tokenizer, model_config=model_config,
                                     eval_batch_size=train_config.eval_batch_size, max_seq_length=args.max_seq_length,
                                     device="cuda:0", use_fixed_seq_length=False, use_tqdm=True)

    result = dict((f"{k}", v) for k, v in result.items())
    print(result)
    eval_task_names = (model_config.task_name,)

    for eval_task_name in eval_task_names:
        result, preds, ex_ids = evaluate(model, tokenizer, model_config=model_config,
                                         eval_batch_size=train_config.eval_batch_size,
                                         device="cuda:0", use_fixed_seq_length=False, use_tqdm=True, split="test",
                                         prefix="")

        processor = processors[eval_task_name]()
        processor.write_preds(preds, ex_ids, "")

    torch.save(model, f"{model_config.model_type}_{model_config.task_name}.pt")


if __name__ == "__main__":
    main()
