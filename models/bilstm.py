import torch
import torch.nn as nn

class BiLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout_prob=0.2):
        super(BiLSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout_prob if num_layers > 1 else 0,
            bidirectional=True
        )

        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        # x shape: [batch_size, seq_len, input_dim]

        lstm_out, (hidden, cell) = self.lstm(x)
        # hidden shape: [num_layers * 2, batch_size, hidden_dim]
        # lstm_out shape: [batch_size, seq_len, hidden_dim * 2]

        # hidden[-2,:,:] -> (forward)
        # hidden[-1,:,:] -> (backward)
        last_hidden_state_forward = hidden[-2,:,:]
        last_hidden_state_backward = hidden[-1,:,:]

        concat_hidden = torch.cat((last_hidden_state_forward, last_hidden_state_backward), dim=1)
        # concat_hidden shape: [batch_size, hidden_dim * 2]

        out = self.dropout(concat_hidden)
        out = self.fc(out) # -> shape: [batch_size, output_dim]

        return out