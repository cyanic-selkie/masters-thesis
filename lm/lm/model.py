from transformers import AutoModelForTokenClassification, AutoTokenizer, AutoConfig, BertPreTrainedModel, BertModel, BertConfig
from typing import List, Optional
import numpy as np
import torch
from typing import Optional, Union, Tuple
import torch.nn as nn
from transformers.utils import ModelOutput

class NELOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    embeddings: torch.FloatTensor
    logits: torch.FloatTensor
   
class NELModel(BertPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.alpha = config.alpha

        # span_indices = torch.linspace(0, config.max_position_embeddings - 1, config.max_position_embeddings, dtype=torch.int32)
        # span_indices = torch.combinations(span_indices, 2, with_replacement=True)
        # self.register_buffer('span_indices', span_indices)

        self.bert = BertModel(config, add_pooling_layer=False)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)

        self.mapper = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.embedding_size),
            nn.GELU(),
            nn.LayerNorm(config.embedding_size, eps=1e-12),
            nn.Linear(config.embedding_size, config.embedding_size),
        )

        self.classifier = nn.Linear(config.embedding_size, 1)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        spans: Optional[torch.Tensor] = None,
        span_mask: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], NELOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)

        # Select the beginning and ending tokens of all spans (note that span boundaries are symmetric).
        # bos = sequence_output[:, self.get_buffer("span_indices")[:, 0], :]
        # eos =  sequence_output[:, self.get_buffer("span_indices")[:, 1], :]
        bos = sequence_output[:, spans[:, 0], :]
        eos =  sequence_output[:, spans[:, 1], :]
        # Combine boundary token embeddings into a single span embedding.
        embeddings = self.mapper(torch.cat((bos, eos), dim=2))
        # Calculate scores for classification.
        logits = self.classifier(embeddings).squeeze()

        loss = None
        losses = []
        if targets is not None:
            loss_fct = nn.MSELoss(reduction='none')

            mse_loss = loss_fct(embeddings, targets)

            # mask
            mask = span_mask.unsqueeze(-1).expand_as(mse_loss)
            mse_loss = (mse_loss * mask.float()).sum() / mask.sum()

            losses.append(self.alpha * mse_loss)

        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss(reduction='none')

            bce_loss = loss_fct(logits, labels)

            # mask
            mask = span_mask
            bce_loss = (bce_loss * mask.float()).sum() / mask.sum()

            losses.append((1 - self.alpha) * bce_loss)

        loss = sum(losses) / len(losses)

        if not return_dict:
            output = (logits, embeddings)
            return ((loss,) + output) if loss is not None else output

        return NELOutput(
            loss=loss,
            embeddings=embeddings,
            logits=logits,
        )

def instantiate_model(checkpoint, embedding_size, alpha):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, model_max_length=512, do_lower_case=False)

    config = AutoConfig.from_pretrained(checkpoint, output_attentions=True, output_hidden_states=True)
    config = config.to_dict()
    config["embedding_size"] = embedding_size
    config["alpha"] = alpha
    config = BertConfig.from_dict(config)

    model = NELModel.from_pretrained(checkpoint, config=config)

    return model, tokenizer
