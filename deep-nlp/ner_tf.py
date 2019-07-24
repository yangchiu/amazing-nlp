import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
from keras.preprocessing.sequence import pad_sequences
import os

save_dir = 'ner_tf/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

training_data_filename = 'ner.txt'


def get_data():
    print(f'* calling {get_data.__name__}')

    word2idx = {}
    tag2idx = {}
    # word_idx starts from 1, reserve idx 0 for pad_sequences
    word_idx = 1
    # tag_idx starts from 1, reserve idx 0 for pad_sequences
    tag_idx = 1

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
            # construct word2idx for this dataset
            if word not in word2idx:
                word2idx[word] = word_idx
                word_idx += 1
            # current_x collect word2idx for this sentence
            current_x.append(word2idx[word])

            # construct tag2idx for this dataset
            if tag not in tag2idx:
                tag2idx[tag] = tag_idx
                tag_idx += 1
            # current_y collect tag2idx for this sentence
            current_y.append(tag2idx[tag])
        # if it's an empty line, it the end of a sentence
        # append them to x_train, y_train
        else:
            x_train.append(current_x)
            y_train.append(current_y)
            current_x = []
            current_y = []

    print(f'=> number of samples: {len(x_train)}')
    print(f'=> number of classes: {len(tag2idx)}')

    x_train, y_train = shuffle(x_train, y_train)
    n_test = int(0.3 * len(x_train))
    x_test = x_train[:n_test]
    y_test = y_train[:n_test]
    x_train = x_train[n_test:]
    y_train = y_train[n_test:]

    print(f'=> number of training data: {len(x_train)}')
    print(f'=> number of test data: {len(x_test)}')

    max_seq_len = max(len(x) for x in x_train + x_test)
    print(f'=> max sentence length = {max_seq_len}')

    # pad sequences
    x_train = pad_sequences(x_train, maxlen=max_seq_len)
    y_train = pad_sequences(y_train, maxlen=max_seq_len)
    x_test = pad_sequences(x_test, maxlen=max_seq_len)
    y_test = pad_sequences(y_test, maxlen=max_seq_len)

    print(f'=> x_train shape: {x_train.shape}')
    print(f'=> y_train shape: {y_train.shape}')

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

    # placeholders
    inputs = tf.placeholder(tf.int32, shape=(None, max_seq_len))
    targets = tf.placeholder(tf.int32, shape=(None, max_seq_len))
    keep_prob = tf.placeholder(tf.float32)

    # embedding
    embedding = tf.Variable(
        np.random.randn(vocab_size, embedding_size) / np.sqrt(vocab_size + embedding_size),
        dtype=tf.float32
    )
    # input shape = (batch_size, steps)
    embedd = tf.nn.embedding_lookup(embedding, inputs)
    # output shape = (batch_size, steps, embedding_size)

    # rnn layer
    cells = []
    for i in range(num_layers):
        cells.append(tf.contrib.rnn.GRUCell(num_units=num_hidden, activation=tf.nn.relu))
    for i in range(num_layers):
        cells[i] = tf.contrib.rnn.DropoutWrapper(cells[i], output_keep_prob=keep_prob)

    multicell = tf.contrib.rnn.MultiRNNCell(cells)

    # input shape = (batch_size, steps, embedding_size)
    outputs, states = tf.nn.dynamic_rnn(multicell, embedd, dtype=tf.float32)
    # output shape = (batch_size, steps, num_hidden)

    # input shape = (batch_size, steps, num_hidden)
    outputs = tf.reshape(outputs, [-1, num_hidden])
    # output shape = (batch_size * steps, num_hidden)
    # so we can perform (W * x + b) to reduce dimension to tag_size

    W = tf.Variable(
        init_weights(num_hidden, tag_size),
        dtype=tf.float32
    )
    b = tf.Variable(
        np.zeros(tag_size),
        dtype=tf.float32
    )
    # input shape = (batch_size * steps, num_hidden)
    logits = tf.nn.xw_plus_b(outputs, W, b)
    # output shape = (batch_size * steps, tag_size)

    # input shape = (batch_size * steps, tag_size)
    logits = tf.reshape(logits, [-1, max_seq_len, tag_size])
    # output shape = (batch_size, steps, tag_size)

    # input shape = (batch_size, steps, tag_size)
    logits_ = tf.argmax(logits, 2)
    # output shape = (batch_size, steps)
    # transform one-hot into numerical

    # loss
    loss = tf.reduce_mean(
        # labels aren't one-hot encoded,
        # so use sparse_softmax_cross_entropy_with_logits
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits,
            labels=targets
        )
    )

    # optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train = optimizer.minimize(loss)

    # init
    init = tf.global_variables_initializer()

    # sess
    with tf.Session() as sess:

        sess.run(init)

        epochs = 8
        batch_size = 32
        n_batches = len(y_train) // batch_size

        for i in range(epochs):

            x_train, y_train = shuffle(x_train, y_train)

            total_loss = 0

            for j in range(n_batches):

                x = x_train[j * batch_size: (j+1) * batch_size]
                y = y_train[j * batch_size: (j+1) * batch_size]

                loss_val, _ = sess.run(
                    [loss, train],
                    feed_dict={
                        inputs: x,
                        targets: y,
                        keep_prob: 0.5
                    }
                )
                total_loss += loss_val

            # after complete an epoch
            predicts = logits_.eval(feed_dict={
                inputs: x_test,
                targets: y_test,
                keep_prob: 1.0
            })
            n_total = 0
            n_correct = 0
            for yi, predicts_i in zip(y_test, predicts):
                yi_labels = yi[yi > 0]
                predicts_i_labels = predicts_i[predicts_i > 0]
                n_correct += np.sum(yi_labels == predicts_i_labels)
                n_total += len(yi_labels)
            print(f'=> epoch {i}, loss = {total_loss}, accuracy = {n_correct/n_total}')


if __name__ == '__main__':

    train_model()
