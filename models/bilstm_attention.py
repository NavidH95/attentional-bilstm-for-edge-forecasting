import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Parameter(torch.rand(hidden_dim))

    def forward(self, hidden, encoder_outputs):
        seq_len = encoder_outputs.size(1)
        hidden = hidden.repeat(seq_len, 1, 1).permute(1, 0, 2)
        energy = torch.tanh(self.attn(torch.cat([hidden, encoder_outputs], dim=2)))
        energy = energy.permute(0, 2, 1)
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)
        attention_scores = torch.bmm(v, energy).squeeze(1)
        weights = torch.softmax(attention_scores, dim=1)
        context = torch.bmm(weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, weights

class BiLSTMAttentionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout_prob=0.2):
        super(BiLSTMAttentionModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True, dropout=dropout_prob if num_layers > 1 else 0)
        self.attention = Attention(hidden_dim * 2)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        encoder_outputs, (hidden, cell) = self.lstm(x)
        last_layer_hidden_forward = hidden[-2,:,:]
        last_layer_hidden_backward = hidden[-1,:,:]
        combined_hidden = torch.cat((last_layer_hidden_forward, last_layer_hidden_backward), dim=1)
        query_for_attention = combined_hidden.unsqueeze(0)
        context_vector, attention_weights = self.attention(query_for_attention, encoder_outputs)
        context_vector = self.dropout(context_vector)
        output = self.fc(context_vector)
        return output, attention_weights