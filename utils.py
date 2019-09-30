import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
import json


def export_result_submit(result_json, file_out, file_template):
    with open(file_out, 'w', encoding='utf-8') as file_out:
        with open(file_template, 'r', encoding='utf-8') as file_in:
            lines = file_in.read().split('\n')
            for line in lines:
                if line.startswith('test_'):
                    key, value = line.split(',')
                    file_out.write("{},{}\n".format(key, result_json[key]))
                elif len(line.strip()) > 0:
                    file_out.write(line + "\n")


def export_result_combine(result_json, file_out):
    with open(file_out, 'w', encoding='utf-8') as file_out:
        json.dump(result_json, file_out, sort_keys=True, indent=2, ensure_ascii=False)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def metrics(y_preds, y, print_report=False, file_log=None):
    """
    Returns accuracy, f1
    """
    # y_preds = np.argmax(preds, axis=-1)
    correct = (y_preds == y)  # convert into float for division
    acc = float(correct.sum()) / len(correct)
    f1 = f1_score(y, y_preds, average=None)
    if print_report:
        print_and_write_log(classification_report(y, y_preds, labels=[0, 1, 2],
                                                  target_names=['CLEAN', 'OFFENSIVE_BUT_NOT_HATE', 'HATE']), file_log)
    return f1, acc


def get_hate_class_f1(f1_value):
    return f1_value[1] * 0.6 + f1_value[2] * 0.4


def get_embedding_dim(embed_type):
    return {
        'comment': 200,
        'comment_bpe': 200,
        'fasttext': 300,
        'sonvx_wiki': 400,
        'sonvx_baomoi_5': 400,
        'sonvx_baomoi_2': 300,
        'roberta': 128,

    }[embed_type]


def print_and_write_log(content, file_log=None):
    print(content)
    if file_log is not None:
        file_log.write(content + "\n")


if __name__ == "__main__":
    print(metrics(np.array([
                               [0.5, 0.1, 0.4],

                           ] * 100000),
                  np.array([0] * 100000), True))
