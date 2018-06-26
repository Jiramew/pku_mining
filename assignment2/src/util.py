import string
import numpy as np


def read_data(filename='../data/train.in', with_filter=True):
    phrase = []

    with open(filename, "r", encoding="utf-8") as fi:
        phrases = fi.read().split("\n")

        for row in phrases:
            if row != "":
                if with_filter:
                    row = "".join(l for l in row if l not in string.punctuation + "’" + "‘").lower()
                phrase.append(row)

    data_array = np.asarray(phrase)

    return data_array


def one_hot(label, factor_num):
    label_one_hot = np.eye(factor_num)[label.reshape(-1)]
    return label_one_hot


def read_glove_vecs(glove_file):
    with open(glove_file, 'r', encoding="utf-8") as f:
        words = set()
        word_to_vec_map = {}
        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            words.add(curr_word)
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)

        i = 1
        words_to_index = {}
        index_to_words = {}
        for w in sorted(words):
            words_to_index[w] = i
            index_to_words[i] = w
            i = i + 1
    return words_to_index, index_to_words, word_to_vec_map


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def average_embedding(sentence, word_vector_map):
    return np.mean(
        [word_vector_map[w] if w in word_vector_map else np.zeros((50,)) for w in
         sentence.lower().split()], axis=0)


def index_embedding(sentence_list, word_to_index, max_len):
    num_od_instance = sentence_list.shape[0]
    embedding = np.zeros((num_od_instance, max_len))
    for i in range(num_od_instance):
        sentence_words = sentence_list[i].lower().split()
        j = 0
        for w in sentence_words:
            if w in word_to_index:
                embedding[i, j] = word_to_index[w]
            j = j + 1
    return embedding


def predict(data, label, weight, bias, word_to_vec_map):
    num_of_instance = data.shape[0]
    pred = np.zeros((num_of_instance, 1))

    for j in range(num_of_instance):
        words = data[j]
        avg = average_embedding(words, word_to_vec_map)

        z = np.dot(weight, avg) + bias
        a = softmax(z)
        pred[j] = np.argmax(a)

    print("Accuracy: " + str(np.mean((pred[:] == label.reshape(label.shape[0], 1)[:]))))
    return pred

