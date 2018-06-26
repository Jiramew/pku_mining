import numpy as np

from keras.models import Model
from keras.layers import Dense, Input, GRU, Activation
from keras.layers.embeddings import Embedding
from src.util import one_hot, softmax, average_embedding, predict


def simple_nn_model(train_data, train_label, word_to_vec_map, learning_rate=0.001, num_iterations=1000):
    num_of_instance = train_label.shape[0]
    label_factor = 2
    num_of_hidden_layer = 50

    weight = np.random.randn(label_factor, num_of_hidden_layer) / np.sqrt(num_of_hidden_layer)
    bias = np.zeros((label_factor,))

    label_one_hot = one_hot(train_label, factor_num=label_factor)

    cost = None
    pred = None

    for t in range(num_iterations):
        for i in range(num_of_instance):
            avg = average_embedding(train_data[i], word_to_vec_map)

            z = np.matmul(weight, avg) + bias
            a = softmax(z)

            cost = -1 * np.sum(label_one_hot[i] * np.log(a))

            grad_z = a - label_one_hot[i]
            grad_weight = np.dot(grad_z.reshape(label_factor, 1), avg.reshape(1, num_of_hidden_layer))
            grad_bias = grad_z

            weight = weight - learning_rate * grad_weight
            bias = bias - learning_rate * grad_bias

        if t % 100 == 0:
            print("Epoch: " + str(t) + " | cost = " + str(cost))
            pred = predict(train_data, train_label, weight, bias, word_to_vec_map)

    return pred, weight, bias


def model_embedding_layer(word_to_vec_map, word_to_index):
    vocab_len = len(word_to_index) + 1
    emb_dim = 50
    emb_matrix = np.zeros((vocab_len, emb_dim))
    for word, index in word_to_index.items():
        emb_matrix[index, :] = word_to_vec_map[word]
    embedding_layer = Embedding(vocab_len, emb_dim, trainable=False)
    embedding_layer.build((None,))
    embedding_layer.set_weights([emb_matrix])

    return embedding_layer


def gru(input_shape, word_to_vec_map, word_to_index):
    # Predicted    0    1  All
    # Actual
    # 0          284   16  300
    # 1           78  222  300
    # All        362  238  600
    sentence_indices = Input(shape=input_shape, dtype='int32')
    embedding_layer = model_embedding_layer(word_to_vec_map, word_to_index)

    embeddings = embedding_layer(sentence_indices)
    # flow = LSTM(256, dropout=0.2, recurrent_dropout=0.2)(embeddings)
    flow = GRU(64, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)(embeddings)
    flow = GRU(64, dropout=0.2, recurrent_dropout=0.2)(flow)
    flow = Dense(2)(flow)
    flow = Activation('softmax')(flow)
    model = Model(inputs=sentence_indices, output=flow)

    return model
