import sys

sys.path.append("../../assignment2")

import numpy as np
import pandas as pd

from src.preprocess import get_data
from src.util import index_embedding, predict
from src.model import simple_nn_model, gru


def train_with_simple_nn_model():
    train_data, train_label, test_data, test_label, \
    train_label_one_hot, test_label_one_hot, word_to_index, \
    index_to_word, word_to_vec_map, phrase_max_length = get_data()

    # Training
    pred, weight, bias = simple_nn_model(train_data, train_label, word_to_vec_map)

    print("Training set:")
    predict(train_data, train_label, weight, bias, word_to_vec_map)
    print('Test set:')
    pred_test = predict(test_data, test_label, weight, bias, word_to_vec_map)

    # Predict
    print(pd.crosstab(test_label,
                      pred_test.reshape(pred_test.shape[0], ),
                      rownames=['Actual'],
                      colnames=['Predicted'],
                      margins=True))


def train_with_gru():
    train_data, train_label, test_data, test_label, \
    train_label_one_hot, test_label_one_hot, word_to_index, \
    index_to_word, word_to_vec_map, phrase_max_length = get_data()

    model = gru((phrase_max_length,), word_to_vec_map, word_to_index)
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    train_data_index_embedding = index_embedding(train_data, word_to_index, phrase_max_length)
    model.fit(train_data_index_embedding, train_label_one_hot, epochs=50, batch_size=32, shuffle=True)

    test_data_index_embedding = index_embedding(test_data, word_to_index, max_len=phrase_max_length)
    loss, acc = model.evaluate(test_data_index_embedding, test_label_one_hot)
    print("Test accuracy = ", acc)

    # Predict
    pred_test = model.predict(test_data_index_embedding)
    print(pd.crosstab(test_label,
                      np.argmax(pred_test, axis=1).reshape(pred_test.shape[0], ),
                      rownames=['Actual'],
                      colnames=['Predicted'],
                      margins=True))


if __name__ == '__main__':
    train_with_gru()
    train_with_simple_nn_model()
