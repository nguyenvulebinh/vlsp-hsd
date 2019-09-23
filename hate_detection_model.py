import torch
import torch.nn as nn
import torch.optim as optim
from models import sample_model
import time
import load_dataset
import os
from tqdm import tqdm
from sklearn.metrics import f1_score
import utils

os.environ["CUDA_VISIBLE_DEVICES"] = "2"


def metrics(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    # round predictions to the closest integer
    # y_preds = torch.argmax(torch.sigmoid(preds), dim=-1)
    y_preds = torch.argmax(preds, dim=-1)
    correct = (y_preds == y).float()  # convert into float for division
    acc = correct.sum() / len(correct)
    f1 = f1_score(y.detach().cpu().numpy(), y_preds.detach().cpu().numpy(), average='weighted')
    return f1, acc


def train(model, iterator, optimizer, criterion, device):
    epoch_loss = 0
    epoch_acc = 0
    epoch_f1 = 0

    model.train()
    count_step = 0
    for batch_inputs, batch_ouputs in tqdm(iterator):
        input_tensor = torch.tensor(batch_inputs, dtype=torch.float32).to(device)
        output_tensor = torch.tensor(batch_ouputs).to(device)

        optimizer.zero_grad()

        predictions = model(input_tensor).squeeze(1)

        loss = criterion(predictions, output_tensor)

        f1, acc = metrics(predictions, output_tensor)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()
        epoch_f1 += f1.item()
        count_step += 1

    return epoch_loss / count_step, epoch_acc / count_step, epoch_f1 / count_step


def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0
    epoch_f1 = 0
    model.eval()
    count_step = 0
    with torch.no_grad():
        for batch_inputs, batch_ouputs in tqdm(iterator):
            input_tensor = torch.tensor(batch_inputs, dtype=torch.float32).to(device)
            output_tensor = torch.tensor(batch_ouputs).to(device)

            predictions = model(input_tensor).squeeze(1)

            loss = criterion(predictions, output_tensor)

            f1, acc = metrics(predictions, output_tensor)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
            epoch_f1 += f1.item()
            count_step += 1

    return epoch_loss / count_step, epoch_acc / count_step, epoch_f1 / count_step


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


if __name__ == "__main__":
    EMBEDDING_DIM = 300
    N_FILTERS = 100
    FILTER_SIZES = [3, 4, 5, 6]
    OUTPUT_DIM = 3
    DROPOUT = 0.5
    BATCH_SIZE = 64
    type_embed = 'fasttext'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model = sample_model.CNN1d(EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT)
    print(model)

    optimizer = optim.Adam(model.parameters())

    criterion = nn.CrossEntropyLoss()

    model = model.to(device)
    criterion = criterion.to(device)

    N_EPOCHS = 5

    best_valid_loss = float('inf')

    for epoch in range(N_EPOCHS):
        start_time = time.time()
        epoch_iterator = load_dataset.generate_batches_train(BATCH_SIZE, type_embed)
        epoch_iterator_valid = load_dataset.generate_batches_train(BATCH_SIZE, type_embed,
                                                                   ids=load_dataset.valid_data_ids,
                                                                   shuffler=False)
        train_loss, train_acc, train_f1 = train(model, epoch_iterator, optimizer, criterion, device)
        valid_loss, valid_acc, valid_f1 = evaluate(model, epoch_iterator_valid, criterion)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%| Train F1: {train_f1 * 100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%| Train F1: {valid_f1 * 100:.2f}%')

    test_iterator = load_dataset.generate_batches_test(BATCH_SIZE, type_embed)
    results_test = {}
    for batch_ids, batch_inputs in tqdm(test_iterator):
        input_tensor = torch.tensor(batch_inputs, dtype=torch.float32).to(device)

        predictions = model(input_tensor).squeeze(1)
        y_preds = torch.argmax(predictions, dim=-1)
        for key, label in zip(batch_ids, y_preds):
            results_test[key] = label
    # print(results_test)
    utils.export_result_submit(results_test, './submit/submit.csv', './data-bin/05_sample_submission.csv')
