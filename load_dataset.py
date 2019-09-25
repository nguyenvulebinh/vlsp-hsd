import pickle
import random
import numpy as np

train_data = pickle.load(open('./data-bin/train_v1.pkl', 'rb'))
test_data = pickle.load(open('./data-bin/test_v1.pkl', 'rb'))
w2v_dict = pickle.load(open('./data-bin/dict_map_v1.pkl', 'rb'))
train_sample2vec_roberta = pickle.load(open('./data-bin/train_sample2vec.pkl', 'rb'))
test_sample2vec_roberta = pickle.load(open('./data-bin/test_sample2vec.pkl', 'rb'))


def get_valid_data_ids(valid_ratio=0.1, shuffer=False):
    all_keys = list(train_data.keys())
    if shuffer:
        random.shuffle(all_keys)
    valid_data_keys = all_keys[:int(len(all_keys) * valid_ratio)]
    return valid_data_keys


valid_data_ids = get_valid_data_ids(shuffer=False)


def get_sample_representation(sample_id, sample_words, present_type, size):
    if present_type == 'roberta':
        embed_vector = None
        if sample_id.startswith("train_"):
            embed_vector = train_sample2vec_roberta[sample_id]
        elif sample_id.startswith("test_"):
            embed_vector = test_sample2vec_roberta[sample_id]
        if embed_vector.shape[0] < size:
            embed_vector = np.concatenate([embed_vector, np.zeros((size - embed_vector.shape[0], 512))], axis=0)
        else:
            embed_vector = embed_vector[:size]
        return embed_vector

    list_embed = []
    for word in sample_words:
        embed = w2v_dict[present_type][word]
        list_embed.append(embed)
    if len(list_embed) < size:
        list_embed.extend([np.zeros((list_embed[0].shape[0],)) for _ in range(size - len(list_embed))])
    elif len(list_embed) > size:
        list_embed = list_embed[:size]
    return np.stack(list_embed)


def generate_batches_test(batch_size, embed_type, max_size=250):
    """
    :param batch_size:
    :param embed_type:
    :param max_size:
    :return:
    """
    ids = list(test_data.keys())

    print("Load test embedding....")
    for sample_id in ids:
        if test_data[sample_id]['representation'][embed_type]['vec'] is not None:
            break
        test_data[sample_id]['representation'][embed_type]['vec'] = \
            get_sample_representation(sample_id,
                                      test_data[sample_id]['representation'][embed_type]['words'],
                                      embed_type,
                                      max_size)
    print("Total: {} samples".format(len(ids)))

    for ndx in range(0, len(ids), batch_size):
        batch_ids = ids[ndx:min(ndx + batch_size, len(ids))]
        batch_input = [test_data[sample_id]['representation'][embed_type]['vec'] for sample_id in batch_ids]
        yield batch_ids, np.stack(batch_input)


def generate_batches_train_for_combine(batch_size, embed_type, max_size=250):
    """
    :param batch_size:
    :param embed_type:
    :param max_size:
    :return:
    """
    ids = list(train_data.keys())

    print("Load all train embedding....")
    for sample_id in ids:
        if train_data[sample_id]['representation'][embed_type]['vec'] is not None:
            break
        train_data[sample_id]['representation'][embed_type]['vec'] = \
            get_sample_representation(sample_id,
                                      train_data[sample_id]['representation'][embed_type]['words'],
                                      embed_type,
                                      max_size)
    print("Total: {} samples".format(len(ids)))

    for ndx in range(0, len(ids), batch_size):
        batch_ids = ids[ndx:min(ndx + batch_size, len(ids))]
        batch_input = [train_data[sample_id]['representation'][embed_type]['vec'] for sample_id in batch_ids]
        batch_output = np.array([train_data[sample_id]['label'] for sample_id in batch_ids])
        yield batch_ids, np.stack(batch_input), batch_output


def generate_batches_train(batch_size, embed_type, shuffler=True, max_size=250, ids=None):
    """
    :param batch_size:
    :param embed_type:
    :param shuffler:
    :param max_size:
    :param ids: if set, it's valid set
    :return:
    """
    if ids is None:
        print("Load train embedding....")
        ids = list(set(train_data.keys()) - set(valid_data_ids))
    else:
        print("Load valid embedding....")
    if shuffler:
        random.shuffle(ids)

    # get max size for batch padding (250 is cover almost train dataset and all test set)
    if max_size is None:
        item_lengths = [len(train_data[item_id]['representation'][embed_type]['words']) for item_id in ids]
        max_size = max(item_lengths)
        # hist = np.histogram(item_lengths)

    for sample_id in ids:
        if train_data[sample_id]['representation'][embed_type]['vec'] is not None:
            break
        train_data[sample_id]['representation'][embed_type]['vec'] = \
            get_sample_representation(sample_id,
                                      train_data[sample_id]['representation'][embed_type]['words'],
                                      embed_type,
                                      max_size)
    print("Total: {} samples".format(len(ids)))
    for ndx in range(0, len(ids), batch_size):
        batch_ids = ids[ndx:min(ndx + batch_size, len(ids))]
        batch_input = [train_data[sample_id]['representation'][embed_type]['vec'] for sample_id in batch_ids]
        batch_output = np.array([train_data[sample_id]['label'] for sample_id in batch_ids])
        # len(batch_vector)
        yield np.stack(batch_input), batch_output


if __name__ == "__main__":
    print(valid_data_ids)
    epoch_iterator = generate_batches_train(20, 'roberta')
    epoch_iterator_test = generate_batches_test(20, 'roberta')
    for batch_inputs, batch_ouputs in epoch_iterator:
        print(batch_inputs.shape, batch_ouputs)
    for batch_inputs in epoch_iterator_test:
        print(batch_inputs.shape)
    #
    # sample_id = 'train_hwstdsnxrv'
    #
    # print(
    #     train_data[sample_id]['raw'], '\n',
    #     train_data[sample_id]['label'], '\n',
    #     train_data[sample_id]['representation']['comment']['words'], '\n',
    #     'comment embedding', get_sample_representation(train_data[sample_id]['representation']['comment']['words'],
    #                                                    'comment', 200).shape, '\n',
    #     'fasttext embedding', get_sample_representation(train_data[sample_id]['representation']['fasttext']['words'],
    #                                                     'fasttext', 200).shape, '\n',
    #     'sonvx_wiki embedding',
    #     get_sample_representation(train_data[sample_id]['representation']['sonvx_wiki']['words'],
    #                               'sonvx_wiki', 200).shape, '\n',
    #     'sonvx_baomoi_5', get_sample_representation(train_data[sample_id]['representation']['sonvx_baomoi_5']['words'],
    #                                                 'sonvx_baomoi_5', 200).shape, '\n',
    #     'sonvx_baomoi_2', get_sample_representation(train_data[sample_id]['representation']['sonvx_baomoi_2']['words'],
    #                                                 'sonvx_baomoi_2', 200).shape, '\n',
    #
    # )
    #
    # test_id = 'test_vauataawkf'
    #
    # print(
    #     test_data[test_id]['raw'], '\n',
    #     test_data[test_id]['label'], '\n',
    #     test_data[test_id]['representation']['fasttext']['words'], '\n',
    #     'comment embedding', get_sample_representation(test_data[test_id]['representation']['comment']['words'],
    #                                                    'comment', 200).shape, '\n',
    #     'fasttext embedding', get_sample_representation(test_data[test_id]['representation']['fasttext']['words'],
    #                                                     'fasttext', 200).shape, '\n',
    #     'sonvx_wiki embedding', get_sample_representation(test_data[test_id]['representation']['sonvx_wiki']['words'],
    #                                                       'sonvx_wiki', 200).shape, '\n',
    #     'sonvx_baomoi_5', get_sample_representation(test_data[test_id]['representation']['sonvx_baomoi_5']['words'],
    #                                                 'sonvx_baomoi_5', 200).shape, '\n',
    #     'sonvx_baomoi_2', get_sample_representation(test_data[test_id]['representation']['sonvx_baomoi_2']['words'],
    #                                                 'sonvx_baomoi_2', 200).shape, '\n',
    # )
