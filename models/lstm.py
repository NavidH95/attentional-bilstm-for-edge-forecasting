import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout_prob=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout_prob if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        # x shape: [batch_size, seq_len, input_dim]

        lstm_out, (hidden, cell) = self.lstm(x)
        # hidden shape: [num_layers, batch_size, hidden_dim]
        # lstm_out shape: [batch_size, seq_len, hidden_dim]

        last_hidden_state = hidden[-1,:,:] # -> shape: [batch_size, hidden_dim]

        out = self.dropout(last_hidden_state)
        out = self.fc(out) # -> shape: [batch_size, output_dim]

        return out