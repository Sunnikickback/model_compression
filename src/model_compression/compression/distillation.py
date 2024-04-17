from torch.nn import MSELoss, CosineSimilarity
from transformers import AdamW, get_linear_schedule_with_warmup, RobertaForSequenceClassification
from torch.utils.data import RandomSampler
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch

from model_compression.training_utils.datasets import load_and_cache_examples
from model_compression.training_utils.training import evaluate
from model_compression.training_utils.utils import TrainConfig, set_seed, task_metrics, DistilConfig, ModelConfig, output_modes, \
    parse_args, get_model


def distill(train_dataset, teacher_model, student_model, tokenizer,
            model_config: ModelConfig, train_config: TrainConfig, distil_config: DistilConfig, device="cuda:0"):
    student_model.to(device)
    teacher_model.to(device)

    output_mode = output_modes[model_config.task_name]
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=train_config.train_batch_size)

    if train_config.max_steps > 0:
        t_total = train_config.max_steps
        train_config.num_train_epochs = train_config.max_steps // (
                len(train_dataloader) // train_config.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // train_config.gradient_accumulation_steps * train_config.num_train_epochs
    num_warmup_steps = int(train_config.warmup_ratio * t_total)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in student_model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": train_config.weight_decay,
        },
        {"params": [p for n, p in student_model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=train_config.learning_rate, eps=train_config.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=t_total
    )

    teacher_layer_num = teacher_model.config.num_hidden_layers
    student_layer_num = student_model.config.num_hidden_layers

    loss_mse = MSELoss()
    loss_cs = CosineSimilarity(dim=2)
    loss_cs_att = CosineSimilarity(dim=3)

    def soft_cross_entropy(predicts, targets):
        student_likelihood = torch.nn.functional.log_softmax(predicts, dim=-1)
        targets_prob = torch.nn.functional.softmax(targets, dim=-1)
        return (- targets_prob * student_likelihood).sum(dim=-1).mean()

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0

    tr_loss, logging_loss = 0.0, 0.0
    student_model.zero_grad()
    train_iterator = range(epochs_trained, int(train_config.num_train_epochs))

    set_seed(1337)

    for epoch_n in train_iterator:
        tr_att_loss = 0.
        tr_rep_loss = 0.
        tr_cls_loss = 0.
        epoch_iterator = tqdm(train_dataloader, desc=f"Epoch {epoch_n}")
        for step, batch in enumerate(epoch_iterator):

            att_loss = 0.
            rep_loss = 0.

            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            student_model.train()
            batch = tuple(t.to(device) for t in batch)
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}

            if output_mode == "span_classification":
                inputs["spans"] = batch[4]

            inputs["token_type_ids"] = None

            outputs_student = student_model(output_attentions=True, output_hidden_states=True, **inputs)

            teacher_model.eval()
            with torch.no_grad():
                outputs_teacher = teacher_model(output_attentions=True, output_hidden_states=True, **inputs)

            kd_loss = soft_cross_entropy(outputs_student[1], outputs_teacher[1])
            loss = kd_loss
            tr_cls_loss += loss.item()

            if distil_config.state_loss_ratio > 0.0:
                teacher_reps = outputs_teacher[2]
                student_reps = outputs_student[2]

                new_teacher_reps = [teacher_reps[0], teacher_reps[teacher_layer_num]]
                new_student_reps = [student_reps[0], student_reps[student_layer_num]]
                for student_rep, teacher_rep in zip(new_student_reps, new_teacher_reps):
                    if distil_config.state_distill_cs:
                        tmp_loss = 1.0 - loss_cs(student_rep, teacher_rep).mean()
                    else:
                        tmp_loss = loss_mse(student_rep, teacher_rep)
                    rep_loss += tmp_loss
                loss += distil_config.state_loss_ratio * rep_loss
                tr_rep_loss += rep_loss.item()

            if distil_config.att_loss_ratio > 0.0:
                teacher_atts = outputs_teacher[3]
                student_atts = outputs_student[3]

                assert teacher_layer_num == len(teacher_atts)
                assert student_layer_num == len(student_atts)
                assert teacher_layer_num % student_layer_num == 0
                layers_per_block = int(teacher_layer_num / student_layer_num)
                new_teacher_atts = [teacher_atts[i * layers_per_block + layers_per_block - 1]
                                    for i in range(student_layer_num)]

                for student_att, teacher_att in zip(student_atts, new_teacher_atts):
                    student_att = torch.where(student_att <= -1e2, torch.zeros_like(student_att).to(device),
                                              student_att)
                    teacher_att = torch.where(teacher_att <= -1e2, torch.zeros_like(teacher_att).to(device),
                                              teacher_att)
                    tmp_loss = 1.0 - loss_cs_att(student_att, teacher_att).mean()
                    att_loss += tmp_loss

                loss += distil_config.att_loss_ratio * att_loss
                tr_att_loss += att_loss.item()

            if train_config.gradient_accumulation_steps > 1:
                loss = loss / train_config.gradient_accumulation_steps

            loss.backward()

            tr_loss += loss.item()
            epoch_iterator.set_description(f"Epoch {epoch_n} loss: {loss:.3f}")
            if (step + 1) % train_config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(student_model.parameters(), train_config.max_grad_norm)

                optimizer.step()
                scheduler.step()
                student_model.zero_grad()
                global_step += 1

                student_model.eval()

                if global_step % train_config.eval_and_save_steps == 0:
                    results, _, _ = evaluate(student_model, tokenizer, model_config=model_config,
                                             eval_batch_size=train_config.eval_batch_size,
                                             device=device, use_fixed_seq_length=False, use_tqdm=True)
                    print(results[task_metrics[model_config.task_name]])
                student_model.train()

            if 0 < train_config.max_steps <= global_step:
                epoch_iterator.close()
                break

        if 0 < train_config.max_steps <= global_step:
            break

    return global_step, tr_loss / global_step


def main():
    args = parse_args("distillation")
    distil_model, tokenizer, model_config = get_model(args)
    distil_config = DistilConfig.from_args(args)
    train_dataset = load_and_cache_examples(tokenizer, model_config, max_seq_length=args.max_seq_length)
    train_config = TrainConfig().from_args(args)
    teacher_model = torch.load(args.teacher_path)
    _, _ = distill(train_dataset, teacher_model, distil_model, tokenizer, model_config, train_config, distil_config)
    result, preds, ex_ids = evaluate(distil_model, tokenizer, model_config=model_config,
                                     eval_batch_size=train_config.eval_batch_size,
                                     device="cuda:0", use_fixed_seq_length=False, use_tqdm=True)
    result = dict((f"{k}", v) for k, v in result.items())
    print(result)
    torch.save(distil_model, f"distilled_{model_config.model_type}_{model_config.task_name}.pt")


if __name__ == "__main__":
    main()
