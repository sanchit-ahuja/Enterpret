import torch
import torch.nn as nn
from torch.nn.modules import dropout


class BERT_SPC(nn.Module):
    def __init__(self, bert, dropout_prob = 0.5, bert_dim = 768, polarities_dim = 3):
        super().__init__()
        self.bert = bert # Get the transformer bert model
        self.dropout = nn.Dropout(dropout_prob)
        self.dense = nn.Linear(bert_dim, polarities_dim)
    

    def forward(self, inputs):
        txt_bert, segment_bert = inputs[0], inputs[1]
        _, pooled_output = self.bert(txt_bert, token_type_ids = segment_bert, 
        return_dict = False)
        pooled_output = self.dropout(pooled_output)
        logits = self.dense(pooled_output)

        return logits