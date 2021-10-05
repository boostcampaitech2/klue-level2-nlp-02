from dataclasses import dataclass, fields
from typing import Optional, Tuple, Any
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import AutoModel, AutoConfig, RobertaPreTrainedModel
from utils import *
from tqdm import tqdm


class MyModel(RobertaPreTrainedModel):
    def __init__(self, model_name, config):
        super().__init__(config)
        self.backbone = AutoModel.from_pretrained(model_name, config=config)
        self.num_labels = config.num_labels
        self.config = config

        self.dense = nn.Linear(config.hidden_size*4, config.hidden_size)

        self.dropout = nn.Dropout(0.2)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ): # Roberta model의 input 형식과 동일합니다.
        outputs = self.backbone(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        ) # Roberta model의 input 형식과 동일하기 때문에 기존 backbone 모델에 넣으면 원하는 output을 가져올 수 있습니다.

        # output[0] = last hidden state / output[1] = pooler_output / output[2] = all hidden states (if 'output_hidden_states': True) == past_key_value, tuple 형태로 반환    
        all_hidden_states = torch.stack(outputs[2]) # (layers , batch, seq_len, hidden_size)
        concat_pooling_layer = torch.cat(
            (all_hidden_states[-1], all_hidden_states[-2], all_hidden_states[-3], all_hidden_states[-4]), -1)  # (batch, seq_len, hidden_size * 4)
        
        concat_pooling_layer = concat_pooling_layer[:, 0, :] # 모든 layer의 CLS embedding vector / (batch, hidden_size * 4)

        logits = self.dropout(concat_pooling_layer)  # (batch, hidden_size * 4)
        logits = self.dense(logits) # (batch, hidden_size * 4) -> (batch, hidden_size*4)
        logits = torch.tanh(logits)  # (batch, hidden_size*4)
        logits = self.dropout(logits)   # (batch, hidden_size*4)
        logits = self.out_proj(logits)  # (batch, hidden_size*4) -> (batch, num_labels)

        loss = None

        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[3:]
            return ((loss,) + output) if loss is not None else output
        # transformers의 Trainer를 사용하기 위해서는 output을 원하는 모델 형태에 맞게 tuple형태로 반환해야 합니다.
        # AutoModelForSequenceClassification의 경우 loss, logits, outputs.hidden_states, outputs.attentions 입니다.
        # 자세한 내용은 Trainer docs 참고
        return (
            loss,
            logits,
            outputs.hidden_states,
            outputs.attentions,
        )
        
class AddFourClassifierRoberta(RobertaPreTrainedModel):
    def __init__(self, model_name, config):
        super().__init__(config)
        self.backbone = AutoModel.from_pretrained(model_name, config=config)
        self.num_labels = config.num_labels
        self.config = config

        self.wide_dense = nn.Linear(config.hidden_size, config.hidden_size*4)
        self.drop_out = nn.Dropout(0.2)
        self.narrow_dense = nn.Linear(config.hidden_size*4, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        outputs = self.backbone(
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
        cls_embedding = outputs[0][:, 0, :] # (batch, hidden_size)                

        logits = self.drop_out(cls_embedding)  # (batch, hidden_size)
        logits = self.wide_dense(logits) # (batch, hidden_size) -> (batch, hidden_size*4)
        logits = self.narrow_dense(self.drop_out(logits)) # (batch, hidden_size)
        logits = torch.tanh(self.drop_out(logits))  # (batch, hidden_size*4)
        logits = self.out_proj(logits)  # (batch, hidden_size*4) -> (batch, num_labels)

        loss = None

        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return (
            loss,
            logits,
            outputs.hidden_states,
            outputs.attentions,
        )