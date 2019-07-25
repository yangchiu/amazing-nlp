import numpy as np
from sklearn.utils import shuffle
from keras.models import Model
from keras.layers import Input, Dense, Embedding, GRU, Dropout
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.optimizers import Adam
import os

save_dir = 'ner_keras/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

training_data_filename = 'ner.txt'


def get_data():
    print(f'* calling {get_data.__name__}')

    x_train = []
    y_train = []

    current_x = []
    current_y = []

    for line in open(os.path.join(save_dir, training_data_filename)):

        # The rstrip() method removes any trailing characters (characters at the end a string),
        # space is the default trailing character to remove.
        line = line.rstrip()

        # if it's not an empty line, there is a word-tag pair in this sentence
        if line:
            word, tag = line.split()
            word = word.lower()
            # current_x collect words in this sentence
            current_x.append(word)
            # current_y collect tags in this sentence
            current_y.append(tag)
        # if it's an empty line, it's the end of a sentence
        # append them to x_train, y_train
        else:
            x_train.append(current_x)
            y_train.append(current_y)
            current_x = []
            current_y = []

    print(f'=> number of samples: {len(x_train)}')

    x_train, y_train = shuffle(x_train, y_train)
    n_test = int(0.3 * len(x_train))
    x_test = x_train[:n_test]
    y_test = y_train[:n_test]
    x_train = x_train[n_test:]
    y_train = y_train[n_test:]

    print(f'=> number of training data: {len(x_train)}')
    print(f'=> number of test data: {len(x_test)}')

    # construct word2idx using keras tokenizer
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(x_train)
    x_train = tokenizer.texts_to_sequences(x_train)
    x_test = tokenizer.texts_to_sequences(x_test)

    word2idx = tokenizer.word_index

    # construct tag2idx using keras tokenizer
    tokenizer2 = Tokenizer()
    tokenizer2.fit_on_texts(y_train)
    y_train = tokenizer2.texts_to_sequences(y_train)
    y_test = tokenizer2.texts_to_sequences(y_test)

    tag2idx = tokenizer2.word_index
    print(f'=> number of classes: {len(tag2idx)}')

    max_seq_len = max(len(x) for x in x_train + x_test)
    print(f'=> max sentence length = {max_seq_len}')

    # pad sequences
    x_train = pad_sequences(x_train, maxlen=max_seq_len)
    y_train = pad_sequences(y_train, maxlen=max_seq_len)
    x_test = pad_sequences(x_test, maxlen=max_seq_len)
    y_test = pad_sequences(y_test, maxlen=max_seq_len)

    # one-hot encoding y
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    print(f'=> x_train shape: {x_train.shape}')
    print(f'=> y_train shape: {y_train.shape}')
    print(f'=> x_test shape: {x_test.shape}')
    print(f'=> y_test shape: {y_test.shape}')

    return x_train, y_train, x_test, y_test, word2idx, tag2idx, max_seq_len


def init_weights(input_dim, output_dim):
    return np.random.randn(input_dim, output_dim) / np.sqrt(input_dim + output_dim)


def train_model():
    print(f'* calling {train_model.__name__}')

    x_train, y_train, x_test, y_test, word2idx, tag2idx, max_seq_len = get_data()

    vocab_size = len(word2idx) + 1 # vocab size + pad
    tag_size = len(tag2idx) + 1 # number of classes + pad

    # config

    learning_rate = 0.01
    num_hidden = 10
    num_layers = 3
    embedding_size = 64

    # model

    inputs = Input(shape=(max_seq_len,))

    x = Embedding(input_dim=vocab_size,
                  input_length=max_seq_len,
                  output_dim=embedding_size)(inputs)

    for i in range(num_layers):
        x = GRU(units=num_hidden, return_sequences=True)(x)
        x = Dropout(0.5)(x)

    outputs = Dense(units=tag_size, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)

    # loss, optimizer, metrics

    model.compile(
        # targets are one-hot encoded, use categorical_crossentropy.
        loss='categorical_crossentropy',
        optimizer=Adam(lr=learning_rate),
        metrics=['accuracy']
    )

    # batch_size isn't specified, default value = 32
    model.fit(
        x=x_train,
        y=y_train,
        epochs=8,
        validation_data=(x_test, y_test)
    )


if __name__ == '__main__':

    train_model()
