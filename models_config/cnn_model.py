from models_architecture.cnn_simple_model import CNN1d
import utils


def get_model(embed_type):
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    EMBEDDING_DIM = utils.get_embedding_dim(embed_type)
    N_FILTERS = 512
    FILTER_SIZES = [3, 4, 5, 6]
    OUTPUT_DIM = 3
    DROPOUT = 0.5

    model = CNN1d(EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT)

    print(model)

    print('The model has {count_parameters(model):,} trainable parameters')
    return model, 'cnn_{}.pt'.format(embed_type)
