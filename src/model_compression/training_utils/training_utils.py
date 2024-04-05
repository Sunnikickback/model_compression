from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup
import random
import numpy as np
import torch
from model_compression.training_utils.metrics import superglue_compute_metrics
from model_compression.training_utils.datasets import load_and_cache_examples
from model_compression.training_utils.utils import TrainConfig, task_metrics
import os

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
def evaluate(task_name, model, tokenizer, eval_batch_size, device, use_fixed_seq_length, output_mode,
             model_type, data_dir, split="dev", prefix="", use_tqdm=True):

    results = {}
    
    eval_dataset = load_and_cache_examples(task_name, tokenizer, split=split, data_dir=data_dir)
    eval_answers = None
        
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
            inputs = {"input_ids": batch[0][:,:batch_seq_length].contiguous(),
                      "attention_mask": batch[1][:,:batch_seq_length].contiguous(),
                      "labels": batch[3]}
            if output_mode == "span_classification":
                inputs["spans"] = batch[4]
            inputs["token_type_ids"] = (
                batch[2][:,:batch_seq_length].contiguous() if model_type 
                    in ["bert", "xlnet", "albert"] else None
            )  
        else:
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
            if output_mode == "span_classification":
                inputs["spans"] = batch[4]
            inputs["token_type_ids"] = (
                batch[2] if model_type in ["bert", "xlnet", "albert"] else None
            )  

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
    eval_loss = eval_loss / nb_eval_steps
    if output_mode in ["classification", "span_classification"] and task_name not in ["record"]:
        preds = np.argmax(preds, axis=1)
    elif output_mode == "regression":
        preds = np.squeeze(preds)
    if split != "test":
        output_dir = ""
        result = superglue_compute_metrics(task_name, preds, out_label_ids, guids=ex_ids, answers=eval_answers)
        results.update(result)
        output_eval_file = os.path.join(output_dir, prefix, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            for key in sorted(result.keys()):
                writer.write("%s = %s\n" % (key, str(result[key])))
    return results, preds, ex_ids

def train(train_dataset, model, tokenizer, output_mode, model_type, task_name, data_dir, train_config:TrainConfig, device="cuda:0"):

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=train_config.train_batch_size)

    if train_config.max_steps > 0:
        t_total = train_config.max_steps
        num_train_epochs = train_config.max_steps // (len(train_dataloader) // train_config.gradient_accumulation_steps) + 1
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

    optimizer = AdamW(optimizer_grouped_parameters, lr=train_config.learning_rate, eps=train_config.adam_epsilon)
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
            inputs["token_type_ids"] = (
                batch[2] if model_type in ["bert", "xlnet", "albert"] else None
            )  
                
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

    
                if (train_config.eval_and_save_steps > 0 and global_step % train_config.eval_and_save_steps == 0):
                    results, _, _ = evaluate(task_name, model, tokenizer, eval_batch_size=train_config.eval_batch_size,
                                             device=device, use_fixed_seq_length=False, output_mode=output_mode,
                                             model_type = model_type, use_tqdm=True,data_dir=data_dir)
                    print(results[task_metrics[task_name]])
                    
            if train_config.max_steps > 0 and global_step >= train_config.max_steps:
                epoch_iterator.close()
                break
        if train_config.max_steps > 0 and global_step >= train_config.max_steps:
            break

    return global_step, tr_loss / global_step