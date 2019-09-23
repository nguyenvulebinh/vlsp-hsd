import torch.nn as nn
import torch
import torch.nn.functional as F


class CNN1d(nn.Module):
    def __init__(self, embedding_dim, n_filters, filter_sizes, output_dim,
                 dropout):
        super().__init__()

        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim,
                      out_channels=n_filters,
                      kernel_size=fs)
            for fs in filter_sizes
        ])

        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text_vec, softmax=False):
        # text_vec = [batch size, sent len, emb dim]
        # embedded = [batch size, emb dim, sent len]
        embedded = text_vec.permute(0, 2, 1)

        conved = [F.relu(conv(embedded)) for conv in self.convs]

        # conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]

        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]

        # pooled_n = [batch size, n_filters]

        cat = self.dropout(torch.cat(pooled, dim=1))

        # cat = [batch size, n_filters * len(filter_sizes)]
        if softmax:
            return F.softmax(self.fc(cat), dim=1)
        return self.fc(cat)


if __name__ == "__main__":
    EMBEDDING_DIM = 200
    N_FILTERS = 100
    FILTER_SIZES = [3, 4, 5, 6]
    OUTPUT_DIM = 3
    DROPOUT = 0.5
    BATCH_SIZE = 2

    model = CNN1d(EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT)
    print(model)
    input_test = torch.rand((BATCH_SIZE, 250, 200))
    output = model(input_test, softmax=True)
    print(output.detach().numpy())
