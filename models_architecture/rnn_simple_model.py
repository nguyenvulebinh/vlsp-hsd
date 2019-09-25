import torch
import torch.nn as nn
import torch.nn.functional as F


class RNN(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim, n_layers,
                 bidirectional, dropout):
        super().__init__()

        self.rnn = nn.LSTM(embedding_dim,
                           hidden_dim,
                           num_layers=n_layers,
                           bidirectional=bidirectional,
                           dropout=dropout)

        self.fc = nn.Linear(hidden_dim * 2, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text_vec, softmax=False):
        # text = [sent len, batch size]
        text_lengths = torch.full((text_vec.size(0),), text_vec.size(1), dtype=torch.int32)
        # embedded = [sent len, batch size, emb dim]
        embedded = text_vec.transpose(1, 0)

        # pack sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths)

        packed_output, (hidden, cell) = self.rnn(packed_embedded)

        # hidden = [batch size, hid dim * num directions]
        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))

        if softmax:
            return F.softmax(self.fc(hidden), dim=1)

        return self.fc(hidden)


if __name__ == "__main__":
    EMBEDDING_DIM = 200
    HIDDEN_DIM = 256
    OUTPUT_DIM = 3
    N_LAYERS = 2
    BIDIRECTIONAL = True
    DROPOUT = 0.5
    batch_size = 2

    model = RNN(EMBEDDING_DIM,
                HIDDEN_DIM,
                OUTPUT_DIM,
                N_LAYERS,
                BIDIRECTIONAL,
                DROPOUT)

    input_tensor = torch.rand((batch_size, 250, EMBEDDING_DIM))
    output_tensor = model(input_tensor, softmax=True)
    print(output_tensor)
