import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, RobertaPreTrainedModel


class FCLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.1, use_activation=True):
        super(FCLayer, self).__init__()
        self.use_activation = use_activation
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, output_dim)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.dropout(x)
        if self.use_activation:
            x = self.tanh(x)
        return self.linear(x)


class Rroberta(RobertaPreTrainedModel):
    """r-bert의 paper 주소를 같이 넣어주셔도 좋습니다. - 문영기 멘토님
    r-bert 개념을 roberta에 적용.
    subject, object에 해당하는 각각의 entity hidden state를 평균,
    FC layer를 거친 뒤 cls vector와 concatenation하여 softmax.
    return : (loss), logits, (hidden_states), (attentions)
    """

    def __init__(self, model_name, config):
        print("Rroberta !!!")
        super().__init__(config)
        config.num_labels = 30
        dropout_rate = 0.1
        self.robert = AutoModel.from_pretrained(model_name, config=config)
        self.num_labels = config.num_labels

        self.cls_fc_layer = FCLayer(
            config.hidden_size, config.hidden_size, dropout_rate
        )
        self.entity_fc_layer = FCLayer(
            config.hidden_size, config.hidden_size, dropout_rate
        )
        self.label_classifier = FCLayer(
            config.hidden_size * 3,
            config.num_labels,
            dropout_rate=0,
            use_activation=False,
        )

    @staticmethod
    def entity_average(hidden_output, e_mask):
        """
        Average the entity hidden state vectors (H_i ~ H_j)
        :param hidden_output: [batch_size, j-i+1, dim]
        :param e_mask: [batch_size, max_seq_len]
                e.g. e_mask[0] == [0, 0, 0, 1, 1, 1, 0, 0, ... 0]
        :return: [batch_size, dim]
        """
        e_mask_unsqueeze = e_mask.unsqueeze(1)  # [b, 1, j-i+1]
        length_tensor = (e_mask != 0).sum(dim=1).unsqueeze(1)  # [batch_size, 1]

        # [b, 1, j-i+1] * [b, j-i+1, dim] = [b, 1, dim] -> [b, dim]
        sum_vector = torch.bmm(e_mask_unsqueeze.float(), hidden_output).squeeze(1)
        avg_vector = sum_vector.float() / length_tensor.float()  # broadcasting
        return avg_vector

    def forward(self, input_ids, attention_mask, e1_mask, e2_mask, labels=None):
        outputs = self.robert(
            input_ids, attention_mask=attention_mask
        )  # sequence_output, pooled_output, (hidden_states), (attentions)
        sequence_output = outputs[0]
        pooled_output = outputs[1]  # [CLS]

        # Average
        e1_h = self.entity_average(sequence_output, e1_mask)
        e2_h = self.entity_average(sequence_output, e2_mask)

        # Dropout -> tanh -> fc_layer (Share FC layer for e1 and e2)
        pooled_output = self.cls_fc_layer(pooled_output)
        e1_h = self.entity_fc_layer(e1_h)
        e2_h = self.entity_fc_layer(e2_h)

        # Concat -> fc_layer
        concat_h = torch.cat([pooled_output, e1_h, e2_h], dim=-1)
        logits = self.label_classifier(concat_h)

        outputs = (logits,) + outputs[
            2:
        ]  # add hidden states and attention if they are here

        # Softmax
        if labels is not None:
            if self.num_labels == 1:
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            outputs = (loss,) + outputs

        return outputs


class ConcatFourClsModel(RobertaPreTrainedModel):
    def __init__(self, model_name, config):
        super().__init__(config)
        print("ConcatFourClsModel !!!")
        self.backbone = AutoModel.from_pretrained(model_name, config=config)
        self.num_labels = config.num_labels
        self.config = config

        self.dense = nn.Linear(config.hidden_size * 4, config.hidden_size)

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
    ):  # Roberta model의 input 형식과 동일합니다.
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
        )  # Roberta model의 input 형식과 동일하기 때문에 기존 backbone 모델에 넣으면 원하는 output을 가져올 수 있습니다.

        # output[0] = last hidden state / output[1] = pooler_output / output[2] = all hidden states (if 'output_hidden_states': True) == past_key_value, tuple 형태로 반환
        all_hidden_states = torch.stack(
            outputs[2]
        )  # (layers , batch, seq_len, hidden_size)
        concat_pooling_layer = torch.cat(
            (
                all_hidden_states[-1],
                all_hidden_states[-2],
                all_hidden_states[-3],
                all_hidden_states[-4],
            ),
            -1,
        )  # (batch, seq_len, hidden_size * 4)

        concat_pooling_layer = concat_pooling_layer[
            :, 0, :
        ]  # 모든 layer의 CLS embedding vector / (batch, hidden_size * 4)

        logits = self.dropout(concat_pooling_layer)  # (batch, hidden_size * 4)
        logits = self.dense(
            logits
        )  # (batch, hidden_size * 4) -> (batch, hidden_size*4)
        logits = torch.tanh(logits)  # (batch, hidden_size*4)
        logits = self.dropout(logits)  # (batch, hidden_size*4)
        logits = self.out_proj(logits)  # (batch, hidden_size*4) -> (batch, num_labels)

        loss = None

        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (
                    labels.dtype == torch.long or labels.dtype == torch.int
                ):
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
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
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
        print("RobertaPreTrainedModel !!!")
        self.backbone = AutoModel.from_pretrained(model_name, config=config)
        self.num_labels = config.num_labels
        self.config = config

        self.wide_dense = nn.Linear(config.hidden_size, config.hidden_size * 4)
        self.drop_out = nn.Dropout(0.2)
        self.narrow_dense = nn.Linear(config.hidden_size * 4, config.hidden_size)
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
        cls_embedding = outputs[0][:, 0, :]  # (batch, hidden_size)

        logits = self.drop_out(cls_embedding)  # (batch, hidden_size)
        logits = self.wide_dense(
            logits
        )  # (batch, hidden_size) -> (batch, hidden_size*4)
        logits = self.narrow_dense(self.drop_out(logits))  # (batch, hidden_size)
        logits = self.drop_out(torch.tanh(logits))  # (batch, hidden_size*4)
        logits = self.out_proj(logits)  # (batch, hidden_size*4) -> (batch, num_labels)

        loss = None

        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (
                    labels.dtype == torch.long or labels.dtype == torch.int
                ):
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
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
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


class AddLayerNorm(RobertaPreTrainedModel):
    def __init__(self, model_name, config):
        super().__init__(config)
        print("AddLayerNorm PreTrainedModel !!!")
        self.backbone = AutoModel.from_pretrained(model_name, config=config)
        self.num_labels = config.num_labels
        self.config = config

        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.wide_dense = nn.Linear(config.hidden_size, config.hidden_size * 4)
        self.drop_out = nn.Dropout(0.2)
        self.narrow_dense = nn.Linear(config.hidden_size * 4, config.hidden_size * 2)
        self.last_dense = nn.Linear(config.hidden_size * 2, config.hidden_size)
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
        cls_embedding = outputs[0][:, 0, :]  # (batch, hidden_size)

        logits = self.wide_dense(
            self.drop_out(cls_embedding)
        )  # (batch, hidden_size) -> (batch, hidden_size*4)
        logits = self.narrow_dense(
            self.drop_out(logits)
        )  # (batch, hidden_size*4) -> (batch, hidden_size*2)
        logits = self.last_dense(
            self.drop_out(logits)
        )  # (batch, hidden_size*2) -> (batch, hidden_size)
        logits = self.layer_norm(logits + cls_embedding)
        logits = self.drop_out(torch.tanh(logits))  # (batch, hidden_size)
        logits = self.out_proj(logits)  # (batch, hidden_size) -> (batch, num_labels)

        loss = None

        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (
                    labels.dtype == torch.long or labels.dtype == torch.int
                ):
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
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
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
