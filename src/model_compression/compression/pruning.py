from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm
import numpy as np
import torch
from heapq import heappush, heappop
from ..training_utils.metrics import superglue_compute_metrics
from ..training_utils.datasets import load_and_cache_examples


def sort_by_importance(weight, bias, importance, num_instances, stride):
    importance_ordered = []
    i = 0
    for heads in importance:
        heappush(importance_ordered, (-heads, i))
        i += 1
    sorted_weight_to_concat = None
    sorted_bias_to_concat = None
    i = 0
    while importance_ordered and i < num_instances:
        head_to_add = heappop(importance_ordered)[1]
        if sorted_weight_to_concat is None:
            sorted_weight_to_concat = (weight.narrow(0, int(head_to_add * stride), int(stride)), )
        else:
            sorted_weight_to_concat += (weight.narrow(0, int(head_to_add * stride), int(stride)), )
        if bias is not None:
            if sorted_bias_to_concat is None:
                sorted_bias_to_concat = (bias.narrow(0, int(head_to_add * stride), int(stride)), )
            else:
                sorted_bias_to_concat += (bias.narrow(0, int(head_to_add * stride), int(stride)), )
        i += 1
    return torch.cat(sorted_weight_to_concat), torch.cat \
        (sorted_bias_to_concat) if sorted_bias_to_concat is not None else None


def prune_rewire(prune_config, task_name, model, tokenizer, output_mode, data_dir, prefix="", use_tqdm=True, device="cpu" ,):
    split = "dev"
    results = {}

    eval_dataset = load_and_cache_examples(task_name, tokenizer, data_dir, split=split)
    eval_answers = None

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=prune_config.eval_batch_size)

    inter_weights = torch.zeros(model.config.num_hidden_layers, model.config.intermediate_size, model.config.hidden_size).to(device)
    inter_biases = torch.zeros(model.config.num_hidden_layers, model.config.intermediate_size).to(device)
    output_weights = torch.zeros(model.config.num_hidden_layers, model.config.hidden_size, model.config.intermediate_size).to(device)

    layers = model.base_model.encoder.layer
    head_importance = torch.zeros(model.config.num_hidden_layers, model.config.num_attention_heads).to(device)
    ffn_importance = torch.zeros(model.config.num_hidden_layers, model.config.intermediate_size).to(device)
    for layer_num in range(model.config.num_hidden_layers):
        inter_weights[layer_num] = layers._modules[str(layer_num)].intermediate.dense.weight.detach().to(device)
        inter_biases[layer_num] = layers._modules[str(layer_num)].intermediate.dense.bias.detach().to(device)
        output_weights[layer_num] = layers._modules[str(layer_num)].output.dense.weight.detach().to(device)

    head_mask = torch.ones(model.config.num_hidden_layers, model.config.num_attention_heads).to(device)
    head_mask.requires_grad_(requires_grad=True)

    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    ex_ids = None
    eval_dataloader = tqdm(eval_dataloader, desc="Evaluating") if use_tqdm else eval_dataloader
    tot_tokens = 0.0
    for batch in eval_dataloader:
        model.eval()
        batch = tuple(t.to(device) for t in batch)
        guids = batch[-1]

        inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
        if output_mode == "span_classification":
            inputs["spans"] = batch[4]

        inputs["token_type_ids"] = None
        outputs = model(output_attentions=True, **inputs, head_mask=head_mask)
        tmp_eval_loss, logits = outputs[:2]

        eval_loss += tmp_eval_loss.mean().item()

        tmp_eval_loss.backward()

        head_importance += head_mask.grad.abs().detach()

        for layer_num in range(model.config.num_hidden_layers):
            ffn_importance[layer_num] += torch.abs(
                torch.sum
                    (layers._modules[str(layer_num)].intermediate.dense.weight.grad.detach( ) *inter_weights[layer_num], 1)
                + layers._modules[str(layer_num)].intermediate.dense.bias.grad.detach( ) *inter_biases[layer_num])

        tot_tokens += inputs["attention_mask"].float().detach().sum().data

        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs["labels"].detach().cpu().numpy()
            ex_ids = [guids.detach().cpu().numpy()]
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)
            ex_ids.append(guids.detach().cpu().numpy())

    head_importance /= tot_tokens

    if not prune_config.dont_normalize_importance_by_layer:
        exponent = 2
        norm_by_layer = torch.pow(torch.pow(head_importance, exponent).sum(-1), 1 / exponent)
        head_importance /= norm_by_layer.unsqueeze(-1) + 1e-20

    head_importance = head_importance.cpu()
    ffn_importance = ffn_importance.cpu()
    num_heads = model.config.num_attention_heads
    head_size = model.config.hidden_size / num_heads
    for layer_num in range(model.config.num_hidden_layers):

        query_weight = layers._modules[str(layer_num)].attention.self.query.weight
        query_bias = layers._modules[str(layer_num)].attention.self.query.bias
        key_weight = layers._modules[str(layer_num)].attention.self.key.weight
        key_bias = layers._modules[str(layer_num)].attention.self.key.bias
        value_weight = layers._modules[str(layer_num)].attention.self.value.weight
        value_bias = layers._modules[str(layer_num)].attention.self.value.bias

        query_weight, query_bias = sort_by_importance(query_weight,
                                                      query_bias,
                                                      head_importance[layer_num],
                                                      prune_config.target_num_heads,
                                                      head_size)
        layers._modules[str(layer_num)].attention.self.query.weight = torch.nn.Parameter(query_weight)
        layers._modules[str(layer_num)].attention.self.query.bias = torch.nn.Parameter(query_bias)
        key_weight, key_bias = sort_by_importance(key_weight,
                                                  key_bias,
                                                  head_importance[layer_num],
                                                  prune_config.target_num_heads,
                                                  head_size)
        layers._modules[str(layer_num)].attention.self.key.weight = torch.nn.Parameter(key_weight)
        layers._modules[str(layer_num)].attention.self.key.bias = torch.nn.Parameter(key_bias)
        value_weight, value_bias = sort_by_importance(value_weight,
                                                      value_bias,
                                                      head_importance[layer_num],
                                                      prune_config.target_num_heads,
                                                      head_size)
        layers._modules[str(layer_num)].attention.self.value.weight = torch.nn.Parameter(value_weight)
        layers._modules[str(layer_num)].attention.self.value.bias = torch.nn.Parameter(value_bias)

        weight_sorted, _ = sort_by_importance(
            layers._modules[str(layer_num)].attention.output.dense.weight.transpose(0, 1),
            None,
            head_importance[layer_num],
            prune_config.target_num_heads,
            head_size)
        weight_sorted = weight_sorted.transpose(0, 1)
        layers._modules[str(layer_num)].attention.output.dense.weight = torch.nn.Parameter(weight_sorted)

        weight_sorted, bias_sorted = sort_by_importance(
            layers._modules[str(layer_num)].intermediate.dense.weight,
            layers._modules[str(layer_num)].intermediate.dense.bias,
            ffn_importance[layer_num],
            prune_config.target_ffn_dim,
            1)
        layers._modules[str(layer_num)].intermediate.dense.weight = torch.nn.Parameter(weight_sorted)
        layers._modules[str(layer_num)].intermediate.dense.bias = torch.nn.Parameter(bias_sorted)


        weight_sorted, _ = sort_by_importance(
            layers._modules[str(layer_num)].output.dense.weight.transpose(0, 1),
            None,
            ffn_importance[layer_num],
            prune_config.target_ffn_dim,
            1)
        weight_sorted = weight_sorted.transpose(0, 1)
        layers._modules[str(layer_num)].output.dense.weight = torch.nn.Parameter(weight_sorted)

    torch.save(model,
               f"pruned_models/pruned_{prune_config.target_num_heads}_{prune_config.target_ffn_dim}_{task_name}.pt")

    ex_ids = np.concatenate(ex_ids, axis=0)
    if output_mode in ["classification", "span_classification"]:
        preds = np.argmax(preds, axis=1)
    elif output_mode == "regression":
        preds = np.squeeze(preds)
    if split != "test":
        result = superglue_compute_metrics(task_name, preds, out_label_ids, guids=ex_ids, answers=eval_answers)
        results.update(result)
        output_eval_file = "eval_results.txt"
        with open(output_eval_file, "w") as writer:
            for key in sorted(result.keys()):
                writer.write("%s = %s\n" % (key, str(result[key])))

    return results, preds, ex_ids


def main():
    pass


if __name__ == "__main__":
    main()
