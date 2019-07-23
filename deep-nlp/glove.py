import numpy as np
import os
import tensorflow as tf
from sklearn.metrics.pairwise import pairwise_distances

from brown_corpus import get_sentences_with_word2idx_limit_vocab

save_dir = 'glove/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

cc_matrix_filename = 'cc_matrix.npy'
model_filename = 'glove_model.npz'
word2idx_filename = 'glove_word2idx.json'


# global vectors for word representation
# using matrix factorization to find Co-occurrence Matrix = W x U
# the dimension of W and U is much smaller than the original Co-occurrence Matrix
# and the W and U is the word vectors we want
class Glove:
    #  context_size    context_size
    # |<---------- word ---------->|
    def __init__(self, embedding_size, vocab_size, context_size):
        print(f'* calling {Glove.__init__.__name__}')

        self.embedding_size = embedding_size
        self.vocab_size = vocab_size
        self.context_size = context_size

        # cc_matrix = co-occurrence matrix
        self.cc_matrix = None

    def build_cc_matrix(self, indexed_sents):
        print(f'* calling {Glove.build_cc_matrix.__name__}')

        n_sents = len(indexed_sents)
        print(f'=> number of sentences to process: {n_sents}')

        cc_matrix = np.zeros((self.vocab_size, self.vocab_size))

        for i, indexed_sent in enumerate(indexed_sents):
            if i % 10000 == 0:
                print(f'=> processed {i}/{n_sents} sentences')
            n = len(indexed_sent)
            for j, word_index in enumerate(indexed_sent):
                start = max(0, j - self.context_size)
                end = min(n, j + self.context_size)

                # cc_matrix for <sos> token
                if j - self.context_size < 0:
                    points = 1 / (j + 1)
                    # word index 0 is for <sos>
                    cc_matrix[word_index, 0] += points
                    cc_matrix[0, word_index] += points

                # cc_matrix for <eos> token
                if j + self.context_size > n:
                    points = 1 / ((n + 1) - j)
                    # word index 1 is for <eos>
                    cc_matrix[word_index, 1] += points
                    cc_matrix[1, word_index] += points

                # left side
                for k in range(start, j):
                    context_word_index = indexed_sent[k]
                    points = 1 / (j - k)
                    cc_matrix[word_index, context_word_index] += points
                    cc_matrix[context_word_index, word_index] += points

                # right side
                for k in range(j + 1, end):
                    context_word_index = indexed_sent[k]
                    points = 1 / (k - j)
                    cc_matrix[word_index, context_word_index] += points
                    cc_matrix[context_word_index, word_index] += points

        self.cc_matrix = cc_matrix
        np.save(os.path.join(save_dir, cc_matrix_filename), cc_matrix)

    def fit(self, indexed_sents, learning_rate=1e-4, reg=0.1, cc_matrix_threshold=100, alpha=0.75, epochs=200):
        print(f'* calling {Glove.fit.__name__}')

        if os.path.exists(os.path.join(save_dir, cc_matrix_filename)):
            print(f'=> use existing cc matrix')
            self.cc_matrix = np.load(os.path.join(save_dir, cc_matrix_filename))
        else:
            print(f'=> build new cc matrix')
            self.build_cc_matrix(indexed_sents)

        # weighted cc_matrix
        m = self.cc_matrix
        # np.zeros: shape should be int or tuple
        weighted_cc = np.zeros((self.vocab_size, self.vocab_size))
        weighted_cc[m < cc_matrix_threshold] = \
            (m[m < cc_matrix_threshold] / cc_matrix_threshold) ** alpha
        weighted_cc[m >= cc_matrix_threshold] = 1

        # target
        log_cc = np.log(m + 1)

        # variables

        # for tf.Vatiable, directly provide "initial_value"
        # which can be a Tensor, or Python object convertible to a Tensor (numpy array)
        # no need to provide "shape"
        W = tf.Variable(
            np.random.randn(self.vocab_size, self.embedding_size) / np.sqrt(self.vocab_size + self.embedding_size),
            dtype=tf.float32
        )
        b = tf.Variable(
            # for tf.zeros, shape can be:
            # (1) a list of integers
            # (2) a tuple of integers
            # (3) or a 1-D Tensor of type int32.
            tf.zeros(self.vocab_size),
            dtype=tf.float32
        )
        U = tf.Variable(
            np.random.randn(self.vocab_size, self.embedding_size) / np.sqrt(self.vocab_size + self.embedding_size),
            dtype=tf.float32
        )
        c = tf.Variable(
            tf.zeros(self.vocab_size),
            dtype=tf.float32
        )

        # constants

        # for tf.constant, directly provide "value",
        # which can be a constant value, or a list of values of type dtype
        # or
        # provide shape like below:
        # tensor = tf.constant(-1.0, shape=[2, 3]) => [[-1. -1. -1.]
        #                                              [-1. -1. -1.]]
        mu = tf.constant(
            log_cc.mean(),
            dtype=tf.float32
        )

        # placeholders

        # for tf.placeholder, shape can be
        # (1) a list of integers
        # (2) a tuple of integers
        # (3) or a 1-D Tensor of type int32.
        tf_weighted_cc = tf.placeholder(
            tf.float32,
            shape=(self.vocab_size, self.vocab_size)
        )
        tf_log_cc = tf.placeholder(
            tf.float32,
            shape=(self.vocab_size, self.vocab_size)
        )

        # loss
        delta = tf.matmul(W, tf.transpose(U)) + b + c + mu - tf_log_cc
        loss = tf.reduce_sum(tf_weighted_cc * delta * delta)
        regularized_loss = loss
        for param in (W, U):
            regularized_loss += reg * tf.reduce_sum(param * param)

        # optimizer
        optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.9)
        train = optimizer.minimize(regularized_loss)

        # init
        init = tf.global_variables_initializer()

        # session
        with tf.Session() as sess:

            sess.run(init)

            for i in range(epochs):

                c, _ = sess.run(
                    [loss, train],
                    feed_dict={
                        tf_log_cc: log_cc,
                        tf_weighted_cc: weighted_cc
                    }
                )
                print(f'=> epoch {i}, loss = {c}')

            self.W, self.U = sess.run([W, U])
            # savez: save several arrays into a single file in uncompressed .npz format.
            np.savez(os.path.join(save_dir, model_filename), self.W, self.U)


def test_model(word2idx, W, V):
    print(f'* calling {test_model.__name__}')

    idx2word = dict((v, k) for k, v in word2idx.items())

    # there are multiple ways to get the "final" word embedding
    # We = (W + V.T) / 2
    # We = W

    # use We = W here
    # We = (W + V.T) / 2
    We = W
    print(f'****** Model Testing Start ******')

    similar('two', word2idx, idx2word, We)
    similar('that', word2idx, idx2word, We)
    similar('his', word2idx, idx2word, We)
    similar('were', word2idx, idx2word, We)
    similar('all', word2idx, idx2word, We)
    similar('area', word2idx, idx2word, We)
    similar('east', word2idx, idx2word, We)
    similar('himself', word2idx, idx2word, We)
    similar('white', word2idx, idx2word, We)
    similar('man', word2idx, idx2word, We)


def similar(word, word2idx, idx2word, W):

    V, D = W.shape

    if word not in word2idx:
        print(f'Sorry, {w} not in word2idx')
        return

    vec = W[word2idx[word]]

    distances = pairwise_distances(vec.reshape(1, D), W, metric='cosine').reshape(V)
    idx = distances.argsort()[1:7]

    print(f'==> 6 words most closet to "{word}"')
    for i in idx:
        print(idx2word[i], distances[i])


if __name__ == '__main__':

    indexed_sents, word2idx = get_sentences_with_word2idx_limit_vocab()
    idx2word = dict((v, k) for k, v in word2idx.items())

    if not os.path.exists(os.path.join(save_dir, model_filename)):
        print(f'=> train new model')
        embedding_size = 150
        vocab_size = len(word2idx)
        window_size = 10

        model = Glove(embedding_size, vocab_size, window_size)
        model.fit(indexed_sents)

        w = model.W
        v = model.U
    else:
        print(f'=> load pre-trained model')
        npz = np.load(os.path.join(save_dir, model_filename))
        w = npz['arr_0']
        v = npz['arr_1']

    test_model(word2idx, w, v)

