import json
from pathlib import Path
import numpy as np
import random
from tqdm import tqdm
import torch
import utils
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import time

combine_folder = "./submit-combine"
N_EPOCHS = 30
BATCH_SIZE = 128


class Net(nn.Module):
    def __init__(self, input_shape=30, num_class=3, drop=0.3):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_shape, 128)
        self.fc2 = nn.Linear(128, num_class)
        self.dropout = nn.Dropout(drop)

    def forward(self, x):
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


def generate_batches_train(batch_size, features, labels, shuffler=True):
    ids = list(labels.keys())
    if shuffler:
        random.shuffle(ids)

    print("Total train: {} samples".format(len(ids)))
    for ndx in range(0, len(ids), batch_size):
        batch_ids = ids[ndx:min(ndx + batch_size, len(ids))]
        batch_input = [features[sample_id] for sample_id in batch_ids]
        batch_output = np.array([labels[sample_id] for sample_id in batch_ids])
        yield np.stack(batch_input), batch_output


def generate_batches_test(batch_size, features):
    ids = list(features.keys())

    print("Total test: {} samples".format(len(ids)))
    for ndx in range(0, len(ids), batch_size):
        batch_ids = ids[ndx:min(ndx + batch_size, len(ids))]
        batch_input = [features[sample_id] for sample_id in batch_ids]
        yield batch_ids, np.stack(batch_input)


def load_train(list_files_path):
    samples_feature_dict = {}
    samples_label_dict = {}
    for path in list_files_path:
        with open(path, 'r', encoding='utf-8') as file_data:
            meta_data = json.load(file_data)
            for key, value in meta_data.items():
                if samples_feature_dict.get(key) is None:
                    samples_feature_dict[key] = []
                    samples_label_dict[key] = value[3]
                samples_feature_dict[key].append(value[:3])
    for key in samples_feature_dict.keys():
        samples_feature_dict[key] = np.array(samples_feature_dict[key]).reshape(-1)
    return samples_feature_dict, samples_label_dict


def load_test(list_files_path):
    samples_feature_dict = {}
    for path in list_files_path:
        with open(path, 'r', encoding='utf-8') as file_data:
            meta_data = json.load(file_data)
            for key, value in meta_data.items():
                if samples_feature_dict.get(key) is None:
                    samples_feature_dict[key] = []
                samples_feature_dict[key].append(value[:3])
    for key in samples_feature_dict.keys():
        samples_feature_dict[key] = np.array(samples_feature_dict[key]).reshape(-1)
    return samples_feature_dict


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


def do_train(model_instance, train_features, train_label, test_features):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: {}".format(device))
    optimizer = optim.Adam(model_instance.parameters(), weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(weight=torch.Tensor([0.1, 0.4, 0.5]))

    model_instance = model_instance.to(device)
    criterion = criterion.to(device)

    for epoch in range(N_EPOCHS):
        start_time = time.time()
        epoch_iterator = generate_batches_train(BATCH_SIZE, train_features, train_label)

        train_loss, train_acc, train_f1 = train_epoch(model_instance, epoch_iterator, optimizer, criterion, device)

        end_time = time.time()

        epoch_mins, epoch_secs = utils.epoch_time(start_time, end_time)

        print(f'Epoch: {epoch + 1:02} | '
              f'Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | '
              f'Train Acc: {train_acc * 100:.2f}%| '
              f'Train F1 hate_class: {utils.get_hate_class_f1(train_f1) * 100:.2f}%')

    # Test process
    test_iterator = generate_batches_test(BATCH_SIZE, test_features)
    results_test, results_test_prob = test(model_instance, test_iterator, device)
    # Export result
    utils.export_result_submit(results_test, f'./submit/submit_combine.csv', './data-bin/05_sample_submission.csv')


if __name__ == "__main__":
    files_train = []
    files_test = []
    for filename in Path(combine_folder).glob('**/*.prob.json'):
        if '_test.' in str(filename):
            files_test.append(str(filename))
        else:
            files_train.append(str(filename))
    assert len(files_train) == len(files_test)

    train_features_load, train_label_load = load_train(files_train)
    test_features_load = load_test(files_test)
    model_instance = Net()
    do_train(model_instance, train_features_load, train_label_load, test_features_load)
