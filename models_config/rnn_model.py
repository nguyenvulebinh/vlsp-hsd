from models_architecture.rnn_simple_model import RNN
import utils


def get_model(embed_type):
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    embedding_dim = utils.get_embedding_dim(embed_type)
    hidden_dim = 512
    output_dim = 3
    n_layer = 2
    bidirectional = True
    dropout = 0.5

    model = RNN(embedding_dim,
                hidden_dim,
                output_dim,
                n_layer,
                bidirectional,
                dropout)

    print(model)
    print(f'The model has {count_parameters(model):,} trainable parameters')
    return model, f'rnn_{embed_type}.pt'
