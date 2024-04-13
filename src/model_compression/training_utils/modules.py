from torch.nn import CrossEntropyLoss, MSELoss
from transformers import RobertaModel, RobertaPreTrainedModel, RobertaConfig
from transformers.models.roberta.modeling_roberta import ROBERTA_INPUTS_DOCSTRING
import torch.nn as nn
import torch


def add_start_docstrings_to_callable(*docstr):
    def docstring_decorator(fn):
        class_name = ":class:`~transformers.{}`".format(fn.__qualname__.split(".")[0])
        intro = "   The {} forward method, overrides the :func:`__call__` special method.".format(class_name)
        note = r"""

    .. note::
        Although the recipe for forward pass needs to be defined within
        this function, one should call the :class:`Module` instance afterwards
        instead of this since the former takes care of running the
        pre and post processing steps while the latter silently ignores them.
        """
        fn.__doc__ = intro + note + "".join(docstr) + (fn.__doc__ if fn.__doc__ is not None else "")
        return fn

    return docstring_decorator


class RobertaForSpanClassification(RobertaPreTrainedModel):
    config_class = RobertaConfig
    base_model_prefix = "roberta"

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.num_spans = config.num_spans

        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.num_spans * config.hidden_size, config.num_labels)

        self.init_weights()

    @add_start_docstrings_to_callable(ROBERTA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids=None,
        spans=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output = outputs[0]

        span_reps = []
        for batch_spans in spans.transpose(0, 1):

            span_mask = torch.zeros_like(sequence_output)
            for batch_idx, (start, end) in enumerate(batch_spans):
                span_mask[batch_idx, start : end + 1] = 1

            masked_output = sequence_output * span_mask
            span_rep, _ = masked_output.max(dim=1)
            span_reps.append(span_rep)

        span_reps = torch.cat(span_reps, dim=1)
        span_reps = self.dropout(span_reps)
        logits = self.classifier(span_reps)
        outputs = (logits,) + outputs[2:]

        if labels is not None:
            if self.num_labels == 1:
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs
