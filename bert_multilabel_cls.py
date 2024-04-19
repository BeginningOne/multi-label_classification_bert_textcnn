# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

class BertMultiLabelCls(nn.Module):
    def __init__(self, hidden_size, class_num, dropout=0.1, cnn_out_channels=100, cnn_kernel_sizes=(3, 4, 5)):
        super(BertMultiLabelCls, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-chinese")
        self.text_cnn = TextCNN(hidden_size, cnn_out_channels, cnn_kernel_sizes)
        self.fc = nn.Linear(hidden_size + cnn_out_channels * len(cnn_kernel_sizes), class_num)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask, token_type_ids):
        bert_outputs = self.bert(input_ids, attention_mask, token_type_ids)
        pooled_output = bert_outputs[1]
        cnn_output = self.text_cnn(bert_outputs[0])  # passing the output of BERT to TextCNN
        combined_output = torch.cat((pooled_output, cnn_output), dim=1)  # concatenate BERT output and TextCNN output
        logits = self.fc(self.dropout(combined_output))
        return torch.sigmoid(logits)


class TextCNN(nn.Module):
    def __init__(self, embedding_dim, out_channels, kernel_sizes):
        super(TextCNN, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim, out_channels=out_channels, kernel_size=ks)
            for ks in kernel_sizes
        ])

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Change shape from [batch_size, seq_len, embedding_dim] to [batch_size, embedding_dim, seq_len]
        x = [F.relu(conv(x)) for conv in self.convs]  # Apply convolution and ReLU activation
        x = [F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2) for conv_out in x]  # Max pooling
        x = torch.cat(x, 1)  # Concatenate the pooled outputs from different kernel sizes
        return x




