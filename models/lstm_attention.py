class LSTMAttentionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout_prob=0.2):
        super(LSTMAttentionModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout_prob if num_layers > 1 else 0)
        self.attention = Attention(hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        # x shape: [batch_size, seq_len, input_dim]

        # LSTM layer
        encoder_outputs, (hidden, cell) = self.lstm(x)
        # encoder_outputs shape: [batch_size, seq_len, hidden_dim]
        # hidden shape: [num_layers, batch_size, hidden_dim]

        # We use the hidden state from the last LSTM layer for attention
        last_layer_hidden = hidden[-1,:,:].unsqueeze(0) # -> [1, batch_size, hidden_dim]

        # Attention layer
        context_vector, attention_weights = self.attention(last_layer_hidden, encoder_outputs)

        # Apply dropout to the context vector before the final layer
        context_vector = self.dropout(context_vector)

        # Final fully connected layer
        output = self.fc(context_vector) # -> [batch_size, output_dim]

        # # ai-edge-torch requires model to return a single tensor
        # if self.training:
        #     return output, attention_weights
        # else:
        #     return output