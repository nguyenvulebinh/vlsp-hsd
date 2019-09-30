from models_architecture.rnn_keras_model import SARNNKeras, HARNN, LSTMKeras
from models_architecture.cnn_keras_model import VDCNN, TextCNN, LSTMCNN
from keras.callbacks import EarlyStopping, ModelCheckpoint
from models_architecture.util import MetricsF1
import os
import load_dataset
from tqdm import tqdm
import numpy as np
import utils

MODEL_PATH = "./model-bin/"
N_EPOCHS = 20
BATCH_SIZE = 256
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda/lib64/libcudart.so.10.0"
# os.environ["LD_LIBRARY_PATH"] = "$LD_LIBRARY_PATH:/usr/local/cuda-9.0/lib64/"


def get_train_input():
    pass


def get_valid_input():
    pass


def train_model(train_input, train_label, valid_input, valid_label, model, model_name):
    print(model.summary())
    checkpoint = ModelCheckpoint(
        filepath=os.path.join(MODEL_PATH, model_name),
        monitor='val_loss', verbose=1,
        mode='min',
        save_best_only=True
    )
    early = EarlyStopping(monitor='val_loss', mode='min', patience=5)
    callbacks_list = [checkpoint, early, MetricsF1()]

    model.fit(
        train_input, train_label,
        validation_data=(valid_input, valid_label),
        callbacks=callbacks_list,
        epochs=N_EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=True,
        class_weight={0: 0.07, 1: 0.4, 2: 0.53}
    )

    # restore model best
    model.load_weights(os.path.join(MODEL_PATH, model_name))
    return model, model_name


def test_model(iterator_test, iterator_valid_combine, model):
    # infer test data
    results_test = {}
    results_probs = {}
    for batch_ids, batch_inputs in tqdm(iterator_test, desc='Run test....'):
        predictions = model.predict(batch_inputs)
        y_predict_prob = np.asarray(predictions)
        y_predict = np.argmax(y_predict_prob, axis=1)

        for id_ in range(len(batch_ids)):
            results_test[batch_ids[id_]] = y_predict[id_]
            results_probs[batch_ids[id_]] = y_predict_prob[id_].tolist()

    # infer valid combine data
    results_valid_probs = {}
    for batch_ids, batch_inputs, batch_ouputs in tqdm(iterator_valid_combine, desc='Run valid combine'):
        predictions = model.predict(batch_inputs)
        y_predict_prob = np.asarray(predictions)

        for id_ in range(len(batch_ids)):
            results_valid_probs[batch_ids[id_]] = y_predict_prob[id_].tolist() + [int(batch_ouputs[id_])]

    return results_test, results_probs, results_valid_probs


if __name__ == "__main__":
    # list_embeds = ['comment', 'fasttext', 'sonvx_wiki', 'sonvx_baomoi_5', 'sonvx_baomoi_2']
    list_embeds = ['comment', 'fasttext', 'comment_bpe', 'roberta']
    # list_embeds = ['fasttext']
    for embed_item in list_embeds:
        list_models = [
            (LSTMKeras(embed_type=embed_item)),
            (LSTMCNN(embed_type=embed_item)),
            (TextCNN(embed_type=embed_item)),
            (VDCNN(embed_type=embed_item)),
            (SARNNKeras(embed_type=embed_item)),
        ]

        train_input, train_label = load_dataset.get_train_data(embed_item)
        valid_input, valid_label = load_dataset.get_valid_data(embed_item)

        for model_item in list_models:
            model_best, model_name = train_model(train_input, train_label, valid_input, valid_label, *model_item)
            r_test, r_prob, rv_prob = \
                test_model(load_dataset.generate_batches_test(batch_size=BATCH_SIZE, embed_type=embed_item),
                           load_dataset.generate_batches_for_combine(batch_size=BATCH_SIZE, embed_type=embed_item),
                           model_best)
            # Export result
            utils.export_result_combine(r_prob, f'./submit-combine/{model_name}_test.prob.json')
            utils.export_result_combine(rv_prob, f'./submit-combine/{model_name}.prob.json')
            utils.export_result_submit(r_test, f'./submit/submit_{model_name}.csv',
                                       './data-bin/05_sample_submission.csv')


# need more attention
# TextCNN_fasttext.hdf5