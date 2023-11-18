from torch import nn

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        # src shape: [src_len, batch_size]
        embedded = self.dropout(self.embedding(src))
        # embedded shape: [src_len, batch_size, emb_dim]
        outputs, (hidden, cell) = self.rnn(embedded)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input, hidden, cell):
        # input shape: [batch_size]
        # hidden shape: [n_layers * n_directions, batch_size, hid_dim]
        # cell shape: [n_layers * n_directions, batch_size, hid_dim]
        input = input.unsqueeze(0)
        # input shape now: [1, batch_size]
        embedded = self.dropout(self.embedding(input))
        # embedded shape: [1, batch_size, emb_dim]
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        # output shape: [seq_len, batch_size, hid_dim * n_directions]
        prediction = self.fc_out(output.squeeze(0))
        # prediction shape: [batch_size, output_dim]
        return prediction, hidden, cell

