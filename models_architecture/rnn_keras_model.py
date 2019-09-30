from keras.models import Model
from keras.layers import \
    Dense, Embedding, Input, \
    CuDNNGRU, GRU, LSTM, Bidirectional, CuDNNLSTM, \
    GlobalMaxPool1D, GlobalAveragePooling1D, Dropout, \
    Lambda, Concatenate, TimeDistributed
from models_architecture.util import f1
from keras_self_attention import SeqSelfAttention, SeqWeightedAttention
from keras.activations import softmax
from keras_layer_normalization import LayerNormalization
from .net_components import AttLayer, AdditiveLayer
from keras.utils.vis_utils import plot_model
import utils
import keras
import keras_metrics as km


def SARNNKeras(embed_type, maxlen=250, rnn_type=LSTM):
    embed_size = utils.get_embedding_dim(embed_type)
    inp = Input(shape=(maxlen, embed_size))
    x = inp

    x = Bidirectional(rnn_type(256, return_sequences=True))(x)
    x = SeqSelfAttention(
        # attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
        attention_regularizer_weight=1e-4,
    )(x)
    # x = LayerNormalization()(x)
    x = Dropout(0.5)(x)

    x = Bidirectional(rnn_type(256, return_sequences=True))(x)
    x = SeqWeightedAttention()(x)
    # x = LayerNormalization()(x)
    x = Dropout(0.5)(x)

    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(3, activation="softmax")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['sparse_categorical_accuracy', km.sparse_categorical_f1_score()])
    return model, 'SARNNKeras_{}.hdf5'.format(embed_type)


def HARNN(embed_type, maxlen=250, rnn_type=LSTM):
    embed_size = utils.get_embedding_dim(embed_type)
    inp = Input(shape=(maxlen, embed_size))
    x = inp

    word_lstm = Bidirectional(rnn_type(256, return_sequences=True))(x)
    word_att = SeqWeightedAttention()(word_lstm)
    word_att = Dropout(0.5)(word_att)
    sent_encoder = Model(x, word_att)
    # plot_model(sent_encoder, to_file='{}.png'.format("HARNN1"), show_shapes=True, show_layer_names=True)

    doc_input = Input(shape=(3, maxlen))
    doc_encoder = TimeDistributed(sent_encoder)(doc_input)
    sent_lstm = Bidirectional(CuDNNLSTM(256, return_sequences=True))(doc_encoder)
    sent_att = SeqWeightedAttention()(sent_lstm)
    sent_att = Dropout(0.5)(sent_att)
    preds = Dense(3, activation="sigmoid")(sent_att)
    model = Model(inputs=doc_input, outputs=preds)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['sparse_categorical_accuracy', km.sparse_categorical_f1_score()])
    return model, 'HARNN_{}.hdf5'.format(embed_type)


def LSTMKeras(embed_type, maxlen=250, rnn_type=LSTM):
    embed_size = utils.get_embedding_dim(embed_type)
    inp = Input(shape=(maxlen, embed_size))
    x = inp
    x = Bidirectional(rnn_type(512, return_sequences=True))(x)
    x = Dropout(0.3)(x)
    x = Bidirectional(rnn_type(512, return_sequences=True))(x)
    x = Dropout(0.3)(x)
    x = GlobalMaxPool1D()(x)

    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(3, activation="softmax")(x)

    model = Model(inputs=inp, outputs=x)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['sparse_categorical_accuracy', km.sparse_categorical_f1_score()])
    return model, 'LSTMKeras_{}.hdf5'.format(embed_type)
