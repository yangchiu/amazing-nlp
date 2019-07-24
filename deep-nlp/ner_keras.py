import numpy as np
from sklearn.utils import shuffle
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, Dense, Embedding, LSTM, GRU
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.optimizers import Adam
import os

save_dir = 'ner_keras/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

training_data_filename = 'ner.txt'


def get_data():
    print(f'* calling {get_data.__name__}')

    Xtrain = []
    Ytrain = []

    currentX = []
    currentY = []

    for line in open(training_data):
        # remove trailing spaces
        line = line.rstrip()
        if line:
            word, tag = line.split()
            word = word.lower()
            # currentX collect words for this sentence
            currentX.append(word)
            # currentY collect tags for this sentence
            currentY.append(tag)

        Xtrain.append(currentX)
        Ytrain.append(currentY)
        currentX = []
        currentY = []

    print(f'=> number of samples: {len(Xtrain)}')
    Xtrain, Ytrain = shuffle(Xtrain, Ytrain)
    Ntest = int(0.3 * len(Xtrain))
    Xtest = Xtrain[:Ntest]
    Ytest = Ytrain[:Ntest]
    Xtrain = Xtrain[Ntest:]
    Ytrain = Ytrain[Ntest:]

    # construct word2idx using keras tokenizer
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(Xtrain)
    Xtrain = tokenizer.texts_to_sequences(Xtrain)
    Xtest = tokenizer.texts_to_sequences(Xtest)

    word2idx = tokenizer.word_index
    print(f'=> found {len(word2idx)} unique tokens')

    # construct tag2idx using keras tokenizer
    tokenizer2 = Tokenizer()
    tokenizer2.fit_on_texts(Ytrain)
    Ytrain = tokenizer2.texts_to_sequences(Ytrain)
    Ytest = tokenizer2.texts_to_sequences(Ytest)

    tag2idx = tokenizer2.word_index
    print(f'=> found {len(tag2idx)} unique tokens')

    return Xtrain, Ytrain, Xtest, Ytest, word2idx, tag2idx


def init_weights(input_dim, output_dim):
    return np.random.randn(input_dim, output_dim) / np.sqrt(input_dim + output_dim)


def train_model():
    print(f'* calling {train_model.__name__}')

    Xtrain, Ytrain, Xtest, Ytest, word2idx, tag2idx = get_data()

    vocab_size = len(word2idx) + 2 # vocab size + unknown + pad
    num_tags = len(tag2idx) + 1 # number of tags

    # config
    epochs = 5
    learning_rate = 1e-2
    batch_size = 32
    hidden_layer_dim = 10
    embedding_dim = 10
    max_seq_len = max(len(x) for x in Xtrain + Xtest)

    # pad sequences
    Xtrain = pad_sequences(Xtrain, maxlen=max_seq_len)
    Ytrain = pad_sequences(Ytrain, maxlen=max_seq_len)
    Xtest = pad_sequences(Xtest, maxlen=max_seq_len)
    Ytest = pad_sequences(Ytest, maxlen=max_seq_len)
    print(f'=> Xtrain shape: {Xtrain.shape}')
    print(f'=> Ytrain.shape: {Ytrain.shape}')

    # one-hot encoding targets
    Ytrain_onehot = np.zeros((len(Ytrain), max_seq_len, num_tags), dtype='float32')
    for n, sample in enumerate(Ytrain):
        for t, tag in enumerate(sample):
            Ytrain_onehot[n, t, tag] = 1

    Ytest_onehot = np.zeros((len(Ytest), max_seq_len, num_tags), dtype='float32')
    for n, sample in enumerate(Ytest):
        for t, tag in enumerate(sample):
            Ytest_onehot[n, t, tag] = 1

    # keras model
    input = Input(shape=(max_seq_len,))
    x = Embedding(vocab_size, embedding_dim)(input)
    x = GRU(hidden_layer_dim, return_sequences=True)(x)
    output = Dense(num_tags, activation='softmax')(x)

    model = Model(input, output)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(lr=learning_rate),
        metrics=['accuracy']
    )

    r = model.fit(
        Xtrain,
        Ytrain_onehot,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(Xtest, Ytest_onehot)
    )

    print(r.history['loss'])
    print(r.history['acc'])


if __name__ == '__main__':

    train_model()
