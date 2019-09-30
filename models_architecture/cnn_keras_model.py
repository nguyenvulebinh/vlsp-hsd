from keras.models import Model
from keras.layers import \
    Dense, Embedding, Input, \
    Conv1D, MaxPool1D, \
    Dropout, BatchNormalization, \
    Bidirectional, CuDNNLSTM, \
    Concatenate, Flatten, Add
from .util import f1
from .net_components import AdditiveLayer
import keras_metrics as km
import utils


def VDCNN(embed_type, maxlen=250, filter_sizes={2, 3, 4, 5}):
    embed_size = utils.get_embedding_dim(embed_type)
    inp = Input(shape=(maxlen, embed_size))
    x = inp

    conv_ops = []
    for filter_size in filter_sizes:
        conv = Conv1D(256, filter_size, activation='relu')(x)
        pool = MaxPool1D(5)(conv)
        conv_ops.append(pool)

    concat = Concatenate(axis=1)(conv_ops)
    # concat = Dropout(0.1)(concat)
    concat = BatchNormalization()(concat)

    conv_2_main = Conv1D(256, 5, activation='relu', padding='same')(concat)
    conv_2_main = BatchNormalization()(conv_2_main)
    conv_2_main = Conv1D(256, 5, activation='relu', padding='same')(conv_2_main)
    conv_2_main = BatchNormalization()(conv_2_main)
    conv_2 = Add()([concat, conv_2_main])
    conv_2 = MaxPool1D(pool_size=2, strides=2)(conv_2)
    # conv_2 = BatchNormalization()(conv_2)
    # conv_2 = Dropout(0.1)(conv_2)

    conv_3_main = Conv1D(256, 5, activation='relu', padding='same')(conv_2)
    conv_3_main = BatchNormalization()(conv_3_main)
    conv_3_main = Conv1D(256, 5, activation='relu', padding='same')(conv_3_main)
    conv_3_main = BatchNormalization()(conv_3_main)
    conv_3 = Add()([conv_2, conv_3_main])
    conv_3 = MaxPool1D(pool_size=2, strides=2)(conv_3)
    # conv_3 = BatchNormalization()(conv_3)
    # conv_3 = Dropout(0.1)(conv_3)

    flat = Flatten()(conv_3)

    op = Dense(256, activation="relu")(flat)
    op = Dropout(0.5)(op)
    op = BatchNormalization()(op)
    op = Dense(128, activation="relu")(op)
    op = Dropout(0.5)(op)
    op = BatchNormalization()(op)
    op = Dense(3, activation="softmax")(op)

    model = Model(inputs=inp, outputs=op)

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['sparse_categorical_accuracy', km.sparse_categorical_f1_score()])
    return model, 'VDCNN_{}.hdf5'.format(embed_type)


# def TextCNN(embeddingMatrix=None, embed_size=400, max_features=20000, maxlen=100, filter_sizes={2, 3, 4, 5},
#             use_fasttext=False, trainable=True, use_additive_emb=False):
def TextCNN(embed_type, maxlen=250, filter_sizes={2, 3, 4, 5}):
    embed_size = utils.get_embedding_dim(embed_type)
    inp = Input(shape=(maxlen, embed_size))
    x = inp

    conv_ops = []
    for filter_size in filter_sizes:
        conv = Conv1D(512, filter_size, activation='relu')(x)
        pool = MaxPool1D(5)(conv)
        conv_ops.append(pool)

    concat = Concatenate(axis=1)(conv_ops)
    # concat = Dropout(0.1)(concat)
    concat = BatchNormalization()(concat)

    conv_2 = Conv1D(512, 5, activation='relu')(concat)
    conv_2 = MaxPool1D(5)(conv_2)
    conv_2 = BatchNormalization()(conv_2)
    # conv_2 = Dropout(0.1)(conv_2)

    conv_3 = Conv1D(512, 5, activation='relu')(conv_2)
    conv_3 = MaxPool1D(5)(conv_3)
    conv_3 = BatchNormalization()(conv_3)
    # conv_3 = Dropout(0.1)(conv_3)

    flat = Flatten()(conv_3)

    op = Dense(256, activation="relu")(flat)
    op = Dropout(0.5)(op)
    op = BatchNormalization()(op)
    op = Dense(128, activation="relu")(op)
    op = Dropout(0.5)(op)
    op = BatchNormalization()(op)
    op = Dense(3, activation="softmax")(op)

    model = Model(inputs=inp, outputs=op)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['sparse_categorical_accuracy', km.sparse_categorical_f1_score()])
    return model, 'TextCNN_{}.hdf5'.format(embed_type)


def LSTMCNN(embed_type, maxlen=250, filter_sizes={2, 3, 4, 5}):
    embed_size = utils.get_embedding_dim(embed_type)
    inp = Input(shape=(maxlen, embed_size))
    x = inp

    x = Bidirectional(CuDNNLSTM(128, return_sequences=True))(x)

    conv_ops = []
    for filter_size in filter_sizes:
        conv = Conv1D(256, filter_size, activation='relu')(x)
        pool = MaxPool1D(5)(conv)
        conv_ops.append(pool)

    concat = Concatenate(axis=1)(conv_ops)
    concat = Dropout(0.5)(concat)
    concat = BatchNormalization()(concat)

    conv_2 = Conv1D(256, 5, activation='relu')(concat)
    conv_2 = MaxPool1D(5)(conv_2)
    conv_2 = BatchNormalization()(conv_2)
    conv_2 = Dropout(0.5)(conv_2)

    conv_3 = Conv1D(256, 5, activation='relu')(conv_2)
    conv_3 = MaxPool1D(5)(conv_3)
    conv_3 = BatchNormalization()(conv_3)
    conv_3 = Dropout(0.1)(conv_3)

    flat = Flatten()(conv_3)

    op = Dense(256, activation="relu")(flat)
    op = Dropout(0.5)(op)
    op = BatchNormalization()(op)
    op = Dense(128, activation="relu")(op)
    op = Dropout(0.5)(op)
    op = BatchNormalization()(op)
    op = Dense(3, activation="softmax")(op)

    model = Model(inputs=inp, outputs=op)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['sparse_categorical_accuracy', km.sparse_categorical_f1_score()])
    return model, 'LSTMCNN_{}.hdf5'.format(embed_type)
