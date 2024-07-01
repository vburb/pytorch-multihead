"""
Implementation borrowed from transformers package and extended to support multiple prediction heads:
https://github.com/huggingface/transformers/blob/master/src/transformers/models/bert/modeling_bert.py
"""

# pylint: disable=[unused-import]
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.bert.modeling_bert import BertModel, BertPreTrainedModel

LABELS = {
    "category_main": ["cat1", "cat2", "cat3", "cat4", "cat5", ],
    "intent": ["praise", "complaint", "suggestion", "comparison",],
    "sentiment": [-1, 0, 1],
}

LABEL_COUNT = {
    "category_main": len(LABELS["category_main"]),
    "intent": len(LABELS["intent"]),
    "sentiment": len(LABELS["sentiment"]),
}


class EncoderModel(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_category_main = LABEL_COUNT["category_main"]  # Number of labels for the first head
        self.num_intent = LABEL_COUNT["intent"]  # Number of labels for the second head

        self.bert = BertModel(config)
        classifier_dropout = config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        self.dropout = nn.Dropout(classifier_dropout)
        # Define two separate classifiers
        self.head_category_main = nn.Linear(config.hidden_size, self.num_category_main)
        self.head_intent = nn.Linear(config.hidden_size, self.num_intent)

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
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
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

        pooled_output = self.dropout(outputs[1])

        # Get logits for both classifiers
        logits_main = self.head_category_main(pooled_output)
        logits_intent = self.head_intent(pooled_output)

        loss = None
        if labels is not None:
            classifier_loss_fct = CrossEntropyLoss()

            label_main = labels[:, 0]  # in label 0 index is main category
            loss_main = classifier_loss_fct(logits_main, label_main)

            label_intent = labels[:, 1]
            loss_sub = classifier_loss_fct(logits_intent, label_intent)

            loss = loss_main + loss_sub  # Combine losses

        if not return_dict:
            output = (logits_main, label_intent) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=(logits_main, label_intent),  # Return logits as a tuple
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
