import numpy as np
from src.util import read_data, one_hot, read_glove_vecs


def get_data(stopwords_filter=True):
    train_data = read_data("../data/train.in")
    train_label = np.asarray(read_data("../data/train.out"), dtype=int)

    test_data = read_data("../data/test.in")
    test_label = np.asarray(read_data("../data/test.out"), dtype=int)

    if stopwords_filter:
        with open("../data/stopwords.txt", "r", encoding="utf-8") as fi:
            stopwords = fi.read().strip().split("\n")
        train_data = [" ".join([w for w in sen.strip().split() if w not in stopwords]) for sen in train_data]
        test_data = [" ".join([w for w in sen.strip().split() if w not in stopwords]) for sen in test_data]

    train_label_one_hot = one_hot(train_label, 2)
    test_label_one_hot = one_hot(test_label, 2)

    phrase_max_length = max(len(max(np.asarray(train_data), key=lambda x: len(x.split())).split()),
                            len(max(np.asarray(test_data), key=lambda x: len(x.split())).split()))

    word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('../data/glove.6B.50d.txt')

    return np.asarray(train_data), train_label, np.asarray(test_data), test_label, train_label_one_hot, \
        test_label_one_hot, word_to_index, index_to_word, word_to_vec_map, phrase_max_length
