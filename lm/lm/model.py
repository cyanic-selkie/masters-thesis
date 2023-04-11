from transformers import AutoModelForTokenClassification, AutoTokenizer, AutoConfig, BertPreTrainedModel, BertModel
from typing import List, Optional
import numpy as np
import torch
from typing import Optional
import torch.nn as nn
   
class NELModel(BertPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config, embedding_size, alpha):
        super().__init__(config)
        self.alpha = alpha
        
        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.mapper = nn.Linear(config.hidden_size, embedding_size)
        self.classifier = nn.Linear(embedding_size, 1)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        targets=None,
        span_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        breakpoint()
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

        # Generate all combination of indices (i.e., all spans) - not permutations.
        indices = torch.linspace(0, sequence_output.shape[1] - 1, sequence_output.shape[1], dtype=torch.int)
        indices = torch.combinations(indices, 2, with_replacement=True).to(input_ids.device)
        # Select the beginning and ending tokens of all spans (note that span boundaries are symmetric).
        bos = sequence_output[:, indices[:, 0], :]
        eos =  sequence_output[:, indices[:, 1], :]
        # Combine boundary token embeddings into a single span embedding.
        embeddings = self.mapper(bos + eos)
        # Calculate scores for classification.
        scores = self.classifier(embeddings).squeeze()

        loss = None
        if targets is not None:
            loss_fct = nn.MSELoss(reduction='none')

            mse_loss = loss_fct(embeddings, targets)

            # mask
            mask = span_mask.unsqueeze(-1).expand_as(mse_loss)
            mse_loss = (mse_loss * mask.float()).sum() / mask.sum()

            loss = self.alpha * mse_loss

        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss(reduction='none')

            bce_loss = loss_fct(scores, labels)

            # mask
            mask = span_mask
            bce_loss = (bce_loss * mask.float()).sum() / mask.sum()

            bce_loss = (1 - self.alpha) * bce_loss
            if loss:
                loss += bce_loss
            else:
                loss = bce_loss

        return loss, (scores, embeddings)

def instantiate_model(checkpoint, architecture, labels, frozen, embedding_size, alpha):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    config = AutoConfig.from_pretrained(checkpoint, output_attentions=True, output_hidden_states=True)
    model = NELModel.from_pretrained(checkpoint, embedding_size=embedding_size, alpha=alpha)

    if frozen:
        for param in model.bert.parameters():
            param.requires_grad = False

    return (model, tokenizer, config)