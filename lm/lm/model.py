from transformers import AutoModelForTokenClassification, AutoTokenizer, AutoConfig, BertPreTrainedModel, BertModel, BertConfig
from typing import List, Optional
import numpy as np
import torch
from typing import Optional, Union, Tuple
import torch.nn as nn
from transformers.utils import ModelOutput
import os

class ELOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    embeddings: torch.FloatTensor
   
class ELModel(BertPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config, add_pooling_layer=False)

        self.mapper_1 = nn.Linear(config.hidden_size * 2, config.embedding_size)

        self.mapper_2 = nn.Sequential(
            nn.GELU(),
            nn.LayerNorm(config.embedding_size),
            nn.Linear(config.embedding_size, config.embedding_size),
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
        spans: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        return_loss: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], ELOutput]:
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

        bos_indices = spans[:,:,0].unsqueeze(-1).expand(-1, -1, sequence_output.shape[-1])
        eos_indices = spans[:,:,1].unsqueeze(-1).expand(-1, -1, sequence_output.shape[-1])
        bos = torch.gather(sequence_output, 1, bos_indices)
        eos = torch.gather(sequence_output, 1, eos_indices)
        # Combine boundary token embeddings into a single span embedding.
        embeddings = self.mapper_1(torch.cat((bos, eos), dim=-1))
        embeddings = self.mapper_2(embeddings) + embeddings

        loss = None
        if return_loss and targets:
            # Spans with indices == 0 are padding;
            # it's enough to only check the start index.
            mask = (spans[:, :, 0] != 0).unsqueeze(-1).expand_as(embeddings)

            if mask.sum() == 0:
                loss = torch.tensor(0., requires_grad=True)
            else:
                loss_fct = nn.MSELoss(reduction='none')
                loss = loss_fct(embeddings, targets)
                loss = (loss * mask.float()).sum() / mask.sum()

        if not return_dict:
            output = (embeddings, )
            return ((loss,) + output) if loss is not None else output

        return ELOutput(
            loss=loss,
            embeddings=embeddings,
        )

def instantiate_model(checkpoint, embedding_size):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, model_max_length=512, do_lower_case=False)
    config = AutoConfig.from_pretrained(checkpoint, output_attentions=True, output_hidden_states=True)
    config = config.to_dict()
    config["embedding_size"] = embedding_size
    config = BertConfig.from_dict(config)
    model = ELModel.from_pretrained(checkpoint, config=config, ignore_mismatched_sizes=True)

    return model, tokenizer
