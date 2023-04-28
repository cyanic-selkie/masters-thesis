from transformers import AutoModelForTokenClassification, AutoTokenizer, AutoConfig, BertPreTrainedModel, BertModel, BertConfig
from typing import List, Optional
import numpy as np
import torch
from typing import Optional, Union, Tuple
import torch.nn as nn
from transformers.utils import ModelOutput
import os

class NELOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    embeddings: torch.FloatTensor
    logits: torch.FloatTensor
   
class NELModel(BertPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.classes_num = config.classes_num

        self.bert = BertModel(config, add_pooling_layer=False)

        self.mapper_1 = nn.Linear(config.hidden_size * 2, config.embedding_size)

        self.mapper_2 = nn.Sequential(
            nn.GELU(),
            nn.LayerNorm(config.embedding_size),
            nn.Linear(config.embedding_size, config.embedding_size),
        )

        self.classifier = nn.Sequential(
            nn.Linear(config.embedding_size, config.embedding_size * 2),
            nn.GELU(),
            nn.LayerNorm(config.embedding_size * 2),
            nn.Linear(config.embedding_size * 2, config.embedding_size * 2),
            nn.GELU(),
            nn.LayerNorm(config.embedding_size * 2),
            nn.Linear(config.embedding_size * 2, config.classes_num),
        )

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
        span_indices: Optional[torch.Tensor] = None,
        span_mask: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        class_weights: Optional[torch.Tensor] = None,
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

        span_indices = span_indices.squeeze()
        bos = sequence_output[:, span_indices[:, 0].squeeze(), :]
        eos =  sequence_output[:, span_indices[:, 1].squeeze(), :]
        # Combine boundary token embeddings into a single span embedding.
        embeddings = self.mapper_1(torch.cat((bos, eos), dim=-1))
        embeddings = self.mapper_2(embeddings) + embeddings
        # Calculate scores for classification.
        logits = self.classifier(embeddings).squeeze()

        loss = None
        losses = []
        if targets is not None:
            mask = span_mask.unsqueeze(-1).expand_as(embeddings)

            if mask.sum() == 0:
                mse_loss = torch.tensor(0., requires_grad=True)
            else:
                loss_fct = nn.MSELoss(reduction='none')
                mse_loss = loss_fct(embeddings, targets)
                mse_loss = (mse_loss * mask.float()).sum() / mask.sum()

            losses.append(mse_loss)

        if labels is not None:
            mask = span_mask.view(-1)

            if mask.sum() == 0:
                bce_loss = torch.tensor(0., requires_grad=True)
            else:
                class_weights = class_weights.squeeze()
                loss_fct = nn.CrossEntropyLoss(reduction='none', weight=class_weights)
                bce_loss = loss_fct(logits.view(-1, self.classes_num), labels.view(-1))
                bce_loss = (bce_loss * mask.float()).sum() / (class_weights[labels.view(-1)] * mask).sum()

            losses.append(bce_loss)

        loss = sum(losses) / len(losses)

        if not return_dict:
            output = (logits, embeddings)
            return ((loss,) + output) if loss is not None else output

        return NELOutput(
            loss=loss,
            embeddings=embeddings,
            logits=logits,
        )

def instantiate_model(checkpoint, embedding_size, classes_num):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, model_max_length=512, do_lower_case=False)
    config = AutoConfig.from_pretrained(checkpoint, output_attentions=True, output_hidden_states=True)
    config = config.to_dict()
    config["embedding_size"] = embedding_size
    config["classes_num"] = classes_num
    config["id2label"] = {
        0: "O",
        1: "ENTITY"
    }
    config["label2id"] = {
        "O": 0,
        "ENTITY": 1
    }
    config = BertConfig.from_dict(config)
    model = NELModel.from_pretrained(checkpoint, config=config, ignore_mismatched_sizes=True)

    return model, tokenizer
