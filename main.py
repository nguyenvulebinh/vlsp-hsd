import torch.optim as optim
# from models_architecture.rnn_simple_model import RNN
# from models_architecture.cnn_simple_model import CNN1d
from models_config import cnn_model, rnn_model
import torch.nn as nn
import torch
from tqdm import tqdm
import utils
import numpy as np
import time
import load_dataset
import os
import torch.nn.functional as F

MODEL_PATH = "./model-bin/"
N_EPOCHS = 20
BATCH_SIZE = 128
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def train_epoch(model, iterator, optimizer, criterion, device, file_log=None):
    epoch_loss = 0
    model.train()
    count_step = 0
    truth_list = np.array([])
    predict_list = np.array([])
    for batch_inputs, batch_ouputs in tqdm(iterator):
        input_tensor = torch.tensor(batch_inputs, dtype=torch.float32).to(device)
        y_true = torch.tensor(batch_ouputs).to(device)

        optimizer.zero_grad()

        predictions = model(input_tensor).squeeze(1)

        loss = criterion(predictions, y_true)

        loss.backward()
        optimizer.step()

        y_preds = torch.argmax(predictions, dim=-1)
        truth_list = np.concatenate([truth_list, y_true.cpu().numpy()], axis=0)
        predict_list = np.concatenate([predict_list, y_preds.cpu().numpy()], axis=0)
        epoch_loss += loss.item()
        count_step += 1

    f1, acc = utils.metrics(predict_list.astype('int16'),
                            truth_list.astype('int16'),
                            print_report=True,
                            file_log=file_log)
    return epoch_loss / count_step, acc, f1


def evaluate_epoch(model, iterator, criterion, device, file_log=None):
    epoch_loss = 0
    model.eval()
    count_step = 0
    truth_list = np.array([])
    predict_list = np.array([])
    with torch.no_grad():
        for batch_inputs, batch_ouputs in tqdm(iterator):
            input_tensor = torch.tensor(batch_inputs, dtype=torch.float32).to(device)
            y_true = torch.tensor(batch_ouputs).to(device)

            predictions = model(input_tensor).squeeze(1)

            loss = criterion(predictions, y_true)

            y_preds = torch.argmax(predictions, dim=-1)
            truth_list = np.concatenate([truth_list, y_true.cpu().numpy().astype('int16')], axis=0)
            predict_list = np.concatenate([predict_list, y_preds.cpu().numpy().astype('int16')], axis=0)
            epoch_loss += loss.item()
            count_step += 1

    f1, acc = utils.metrics(predict_list,
                            truth_list,
                            print_report=True,
                            file_log=file_log)
    return epoch_loss / count_step, acc, f1


def evaluate_epoch_export_prob(model, iterator, device):
    model.eval()
    results_probs = {}
    with torch.no_grad():
        for batch_ids, batch_inputs, batch_ouputs in tqdm(iterator):
            input_tensor = torch.tensor(batch_inputs, dtype=torch.float32).to(device)

            predictions = model(input_tensor).squeeze(1)
            predictions_prob = F.softmax(predictions, dim=-1)

            for id_ in range(len(batch_ids)):
                results_probs[batch_ids[id_]] = predictions_prob[id_].cpu().numpy().tolist() + [int(batch_ouputs[id_])]

    return results_probs


def test(model, iterator, device):
    results_test = {}
    results_probs = {}
    model.eval()
    with torch.no_grad():
        for batch_ids, batch_inputs in tqdm(iterator):
            input_tensor = torch.tensor(batch_inputs, dtype=torch.float32).to(device)

            predictions = model(input_tensor).squeeze(1)
            predictions_prob = F.softmax(predictions, dim=-1)
            y_preds = torch.argmax(predictions, dim=-1).cpu().numpy().astype('int16').tolist()
            for id_ in range(len(batch_ids)):
                results_test[batch_ids[id_]] = y_preds[id_]
                results_probs[batch_ids[id_]] = predictions_prob[id_].cpu().numpy().tolist()

    return results_test, results_probs


def do_train_model(type_embed, model_instance, model_name):
    file_log = open(f'./train-logs/log_{model_name}.txt', 'w', encoding='utf-8')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    utils.print_and_write_log("Device: {}".format(device), file_log)
    optimizer = optim.Adam(model_instance.parameters(), weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(weight=torch.Tensor([0.1, 0.4, 0.5]))

    model_instance = model_instance.to(device)
    criterion = criterion.to(device)

    best_valid_f1 = -float('inf')
    # Change valid set for other model
    load_dataset.valid_data_ids = load_dataset.get_valid_data_ids(shuffer=True)
    # Train process
    for epoch in range(N_EPOCHS):
        start_time = time.time()
        epoch_iterator = load_dataset.generate_batches_train(BATCH_SIZE, type_embed)
        epoch_iterator_valid = load_dataset.generate_batches_train(BATCH_SIZE, type_embed,
                                                                   ids=load_dataset.valid_data_ids,
                                                                   shuffler=False)
        train_loss, train_acc, train_f1 = train_epoch(model_instance, epoch_iterator, optimizer, criterion, device,
                                                      file_log=file_log)
        valid_loss, valid_acc, valid_f1 = evaluate_epoch(model_instance, epoch_iterator_valid, criterion, device,
                                                         file_log=file_log)

        end_time = time.time()

        epoch_mins, epoch_secs = utils.epoch_time(start_time, end_time)

        if utils.get_hate_class_f1(valid_f1) > best_valid_f1:
            utils.print_and_write_log(f'Current best {model_name} epoch {epoch}. Save model....')
            best_valid_f1 = utils.get_hate_class_f1(valid_f1)
            torch.save(model_instance.state_dict(), os.path.join(MODEL_PATH, model_name))

        utils.print_and_write_log(f'Epoch: {epoch + 1:02} | '
                                  f'Epoch Time: {epoch_mins}m {epoch_secs}s', file_log)
        utils.print_and_write_log(f'\tTrain Loss: {train_loss:.3f} | '
                                  f'Train Acc: {train_acc * 100:.2f}%| '
                                  f'Train F1 hate_class: {utils.get_hate_class_f1(train_f1) * 100:.2f}%', file_log)
        utils.print_and_write_log(f'\t Val. Loss: {valid_loss:.3f} |  '
                                  f'Val. Acc: {valid_acc * 100:.2f}%| '
                                  f'Train F1 hate_class: {utils.get_hate_class_f1(valid_f1) * 100:.2f}%', file_log)

    # Test process
    test_iterator = load_dataset.generate_batches_test(BATCH_SIZE, type_embed)
    model_instance.load_state_dict(torch.load(os.path.join(MODEL_PATH, model_name)))
    results_test, results_test_prob = test(model_instance, test_iterator, device)
    utils.export_result_combine(results_test_prob, f'./submit-combine/{model_name}_test.prob.json')
    # Export result
    utils.export_result_submit(results_test, f'./submit/submit_{model_name}.csv', './data-bin/05_sample_submission.csv')

    # Export probs for sample in train data
    train_iterator_for_combine = load_dataset.generate_batches_train_for_combine(BATCH_SIZE, type_embed)
    results_probs = evaluate_epoch_export_prob(model_instance, train_iterator_for_combine, device)
    utils.export_result_combine(results_probs, f'./submit-combine/{model_name}.prob.json')

    file_log.close()


if __name__ == "__main__":
    # list_embeds = ['comment', 'fasttext', 'sonvx_wiki', 'sonvx_baomoi_5', 'sonvx_baomoi_2']
    list_embeds = ['roberta']
    for embed_item in list_embeds:
        list_models = [
            (cnn_model.get_model(embed_item)),
            (rnn_model.get_model(embed_item)),
        ]
        for model_item in list_models:
            do_train_model(embed_item, *model_item)
