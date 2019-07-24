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

    return x_train, y_train, x_test, y_test, word2idx, tag2idx


def init_weights(input_dim, output_dim):
    return np.random.randn(input_dim, output_dim) / np.sqrt(input_dim + output_dim)


def train_model():
    print(f'* calling {train_model.__name__}')

    x_train, y_train, x_test, y_test, word2idx, tag2idx = get_data()

    V = len(word2idx) + 2 # vocab size + unknown + pad
    K = len(tag2idx) + 1 # number of classes

    # config
    epochs = 5
    learning_rate = 1e-2
    mu = 0.99
    batch_size = 32
    hidden_layer_dim = 10
    embedding_dim = 10
    max_seq_len = max(len(x) for x in Xtrain + Xtest)

    # pad sequences
    Xtrain = tf.keras.preprocessing.sequence.pad_sequences(Xtrain, maxlen=max_seq_len)
    Ytrain = tf.keras.preprocessing.sequence.pad_sequences(Ytrain, maxlen=max_seq_len)
    Xtest = tf.keras.preprocessing.sequence.pad_sequences(Xtest, maxlen=max_seq_len)
    Ytest = tf.keras.preprocessing.sequence.pad_sequences(Ytest, maxlen=max_seq_len)
    print(f'=> Xtrain shape: {Xtrain.shape}')
    print(f'=> Ytrain.shape: {Ytrain.shape}')

    # tensorflow model
    # inputs
    inputs = tf.placeholder(tf.int32, shape=(None, max_seq_len)) # (N x T) x V
    targets = tf.placeholder(tf.int32, shape=(None, max_seq_len)) # (N x T) x K
    num_samples = tf.shape(inputs)[0] # N

    # embedding layer
    We = np.random.randn(V, embedding_dim).astype(np.float32) # V x D

    # output layer
    Wo = init_weight(hidden_layer_dim, K).astype(np.float32)
    bo = np.zeros(K).astype(np.float32)

    tf_We = tf.Variable(We)
    tf_Wo = tf.Variable(Wo)
    tf_bo = tf.Variable(bo)

    rnn_unit = GRUCell(num_units=hidden_layer_dim, activation=tf.nn.relu)

    x = tf.nn.embedding_lookup(tf_We, inputs) # (N x T) x D

    # covert x from N x T x D
    # into a list of length T, where each element is N x D
    x = tf.unstack(x, max_seq_len, 1)

    outputs, states = get_rnn_output(rnn_unit, x, dtype=tf.float32)

    # convert outputs from T x N x hidden_layer_dim
    # into N x T x hidden_layer_dim
    outputs = tf.transpose(outputs, (1, 0, 2))
    # from N x T x hidden_layer_dim
    # into NT x hidden_layer_dim
    outputs = tf.reshape(outputs, (num_samples * max_seq_len, hidden_layer_dim))

    logits = tf.matmul(outputs, tf_Wo) + tf_bo # NT x K
    predictions = tf.argmax(logits, 1) # NT
    predict_op = tf.reshape(predictions, (num_samples, max_seq_len)) # N x T
    labels_flat = tf.reshape(targets, [-1])

    cost_op = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits,
            labels=labels_flat
        )
    )
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(cost_op)

    # init session
    sess = tf.InteractiveSession()
    init = tf.global_variables_initializer()
    sess.run(init)

    # training loop
    costs = []
    n_batches = len(Ytrain) // batch_size
    for i in range(epochs):
        n_total = 0
        n_correct = 0

        Xtrain, Ytrain = shuffle(Xtrain, Ytrain)
        cost = 0

        for j in range(n_batches):
            x = Xtrain[j * batch_size:(j+1) * batch_size]
            y = Ytrain[j * batch_size:(j+1) * batch_size]

            c, p, _ = sess.run(
                (cost_op, predict_op, train_op),
                feed_dict={
                    inputs: x,
                    targets: y
                }
            )
            cost += c

            for yi, pi in zip(y, p):
                yii = yi[yi > 0]
                pii = pi[pi > 0]
                n_correct += np.sum(yii == pii)
                n_total += len(yii)

            if j % 10 == 0:
                print(f'{j}/{n_batches} correct rate: {n_correct/n_total} cost: {cost/j}')

        p = sess.run(predict_op, feed_dict={inputs: Xtest, targets: Ytest})
        n_test_correct = 0
        n_test_total = 0
        for yi, pi in zip(Ytest, p):
            yii = yi[yi > 0]
            pii = pi[pi > 0]
            n_test_correct += np.sum(yii == pii)
            n_test_total += len(yii)
        test_acc = float(n_test_correct/n_test_total)

        print(f'{i} cost: {cost/n_batches} train acc = {n_correct/n_total} test acc: {test_acc}')
        costs.append(cost)

if __name__ == '__main__':
    train_model()
