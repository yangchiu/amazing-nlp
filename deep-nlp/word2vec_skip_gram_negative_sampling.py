# tensorflow-gpu version log:
# windows 10
# cuda 10.0
# cudnn 7.4.1.5
# tensorflow-gpu 1.13.1
import collections
from collections import Counter
import os
import random
import zipfile
from sklearn.metrics.pairwise import pairwise_distances
import numpy as np
import tensorflow as tf
# six is a package that helps in writing code that is compatible with both Python 2 and Python 3.
#
# One of the problems developers face when writing code for Python2 and 3 is that
# the names of several modules from the standard library have changed,
# even though the functionality remains the same.
#
# six provides a consistent interface to them through the fake six.moves module.
from six.moves import urllib
import json

save_dir = 'word2vec_skip_gram_negative_sampling/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

data_dir = os.path.join(save_dir, 'data/')
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

data_url = 'http://mattmahoney.net/dc/text8.zip'

word2idx_filename = 'word2idx.json'
weights_filename = 'weights.npz'

class Text8():

    def __init__(self, data_url, data_dir):

        self.corpus = None
        self.fetch_corpus(data_url, data_dir)

        self.indexed_corpus = None
        self.word2idx = None
        self.idx2word = None
        self.vocab = None
        self.word_counts()

        self.corpus_len = len(self.indexed_corpus)
        self.current_corpus_index = 0

    def fetch_corpus(self, url=data_url, dir=data_dir):
        print(f'* calling {Text8.fetch_corpus.__name__}')

        zip_path = os.path.join(dir, 'text8.zip')

        if not os.path.exists(zip_path):
            print(f'=> text8 corpus not found, download it now ...')
            urllib.request.urlretrieve(url, zip_path)
        else:
            print(f'=> use existing text8 corpus')

        with zipfile.ZipFile(zip_path) as f:
            print(f'=> read text8 corpus from text8.zip ...')

            # have to specify the name of the file to be read
            # use f.namelist() to get filename list in the zip file
            # there's only one file in the zip. it's called "text8"
            self.corpus = f.read(f.namelist()[0])

            # split the corpus by whitspace
            # convert to a list of all words in the corpus
            self.corpus = self.corpus.decode('ascii').split()

    def word_counts(self, vocab_size=50000):
        print(f'* calling {Text8.word_counts.__name__}')

        # Counter.most_common return a list of tuple('word', count)
        self.vocab = Counter(self.corpus).most_common(vocab_size)

        # extract all words into a np.array, we don't care the counts
        self.vocab = np.array([word for word, _ in self.vocab])
        self.vocab = np.append(self.vocab, ['<unk>'])
        print(f'=> most common 16 words in this corpus:')
        print(self.vocab[:16])
        print(f'=> most uncommon 16 words in this corpus:')
        print(self.vocab[-16:])

        self.word2idx = {word: idx for idx, word in enumerate(self.vocab)}

        self.idx2word = dict((v, k) for k, v in self.word2idx.items())

        self.indexed_corpus = np.array([
            self.word2idx.get(word, self.word2idx.get('<unk>')) for word in self.corpus
        ])
        print('=> indexed corpus sample:')
        print(self.indexed_corpus[:50])
        print('=> covert indexed corpus sample back into corpus:')
        print(np.array([self.idx2word.get(idx) for idx in self.indexed_corpus[:50]]))
        print('=> the original corpus:')
        print(self.corpus[:50])

    def next_batch(self, batch_size, num_skips, skip_window):

        # num_skips * int should equal to batch_size
        assert batch_size % num_skips == 0
        # num_skips is the times we sampled in the same window
        # so it should smaller than window size
        assert num_skips <= 2 * skip_window

        # batch = input (x)
        batch = np.ndarray(shape=(batch_size,), dtype=np.int32)

        # labels = outputs (y_true)
        labels = np.ndarray(shape=(batch_size,), dtype=np.int32)

        # span = the whole range can be sampled
        # [ skip_window <-- target --> skip_window ]
        span = 2 * skip_window + 1

        # deque = double-ended queue
        # list-like container with fast appends and pops on either end
        dq = collections.deque(maxlen=span)

        if self.current_corpus_index + span > self.corpus_len:
            self.current_corpus_index = 0
        # feed "words in the range can be sampled" into dq
        dq.extend(self.indexed_corpus[self.current_corpus_index:self.current_corpus_index + span])
        self.current_corpus_index += span

        # totally we need # of batch_size data,
        # but we will sample the same word for # of num_skips times,
        # so we only need batch_size // num_skips times iteration.
        for i in range(batch_size // num_skips):
            # target label at the center of the dq
            # if skip_window = 1
            #         _______ target = 1 = skip_window
            #        |
            # dq [ 0 1 2 ]
            #
            # if skip_window = 2
            #          _______ target = 2 = skip_window
            #         |
            # dq [0 1 2 3 4 ]
            target = skip_window
            # indexes in this list have already been sampled,
            # or it's the target word, not a context word
            targets_to_avoid = [skip_window]
            for j in range(num_skips):
                while target in targets_to_avoid:
                    target = random.randint(0, span - 1)
                targets_to_avoid.append(target)
                # skip_gram
                # use "fixed input word" to predict "context words"
                # so batch[idx] is fixed, labels[idx] is variable
                batch[i * num_skips + j] = dq[skip_window]
                labels[i * num_skips + j] = dq[target]
            # after got # of num_skip sample,
            # move to the next window
            if self.current_corpus_index == self.corpus_len:
                dq.extend(self.indexed_corpus[:span])
                self.current_corpus_index = span
            else:
                dq.append(self.indexed_corpus[self.current_corpus_index])
                self.current_corpus_index += 1
        # shift back a little bit to avoid skipping words in the end of a batch
        self.current_corpus_index = (self.current_corpus_index + self.corpus_len - span) % self.corpus_len
        return batch, labels


def train_model():
    print(f'* calling {train_model.__name__}')

    text8 = Text8(data_url, data_dir)

    # constants
    batch_size = 128
    embedding_size = 128
    # how many words to consider left and right
    # window = [ skip_window <-- target --> skip_window ]
    skip_window = 1
    # how many times to reuse an input (target word) to find a label (context word)
    num_skips = 2

    # number of negative examples to sample in nce_loss
    num_neg_sampled = batch_size

    learning_rate = 1.0
    vocab_size = 50001  # 50000 vocab + <unk>

    epochs = 100000

    print(f'=> word2vec dimension is {embedding_size}')
    print(f'=> build word2vec with vocab size: {vocab_size}')

    # distribution for drawing negative samples
    p_neg = get_negative_sampling_distribution(text8.indexed_corpus, vocab_size)
    print(f'=> p_neg shape {p_neg.shape}')
    print(f'=> p_neg: {p_neg}')
    print(f'=> p_neg max: {p_neg.max()}')
    print(f'=> p_neg min: {p_neg.min()}')

    print(f'=> p_neg for [the] = {p_neg[text8.word2idx["the"]]}')
    print(f'=> p_neg for [of] = {p_neg[text8.word2idx["of"]]}')
    print(f'=> p_neg for [and] = {p_neg[text8.word2idx["and"]]}')
    print(f'=> p_neg for [aspirant] = {p_neg[text8.word2idx["aspirant"]]}')
    print(f'=> p_neg for [fidesz] = {p_neg[text8.word2idx["fidesz"]]}')
    print(f'=> p_neg for [teat] = {p_neg[text8.word2idx["teat"]]}')

    # placeholders
    pos_words = tf.placeholder(tf.int32, shape=(None,)) # x
    neg_words = tf.placeholder(tf.int32, shape=(None,)) # x
    context_words = tf.placeholder(tf.int32, shape=(None,)) # y

    # variables
    W = tf.Variable(
        tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0) # input to hidden
    )
    V = tf.Variable(
        tf.random_uniform([embedding_size, vocab_size], -1.0, 1.0) # hidden to output
    )
    emb_pos = tf.nn.embedding_lookup(W, pos_words)
    emb_neg = tf.nn.embedding_lookup(W, neg_words)
    emb_context = tf.nn.embedding_lookup(tf.transpose(V), context_words)

    def dot(A, B):
        C = A * B
        return tf.reduce_sum(C, axis=1)

    # loss
    pos_results = dot(emb_pos, emb_context)
    pos_loss = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.ones_like(pos_results),
        logits=pos_results
    )

    neg_results = dot(emb_neg, emb_context)
    neg_loss = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.zeros_like(neg_results),
        logits=neg_results
    )

    loss = tf.reduce_mean(pos_loss + neg_loss)

    # optimizer
    optimizer = tf.train.AdagradOptimizer(learning_rate)
    train = optimizer.minimize(loss)

    # init
    init = tf.global_variables_initializer()

    # session
    with tf.Session() as sess:

        sess.run(init)

        for epoch in range(epochs):

            batch_pos, batch_context = text8.next_batch(batch_size=batch_size,
                                                        num_skips=num_skips,
                                                        skip_window=skip_window)
            batch_neg = np.random.choice(vocab_size, size=num_neg_sampled, p=p_neg)

            _, loss_val = sess.run(
                (train, loss),
                feed_dict={
                    pos_words: batch_pos,
                    neg_words: batch_neg,
                    context_words: batch_context,
                }
            )

            if epoch % 1000 == 0:
                print(f'=> {epoch} / {epochs}, loss = {loss_val}')

        W_eval = W.eval()
        V_eval = V.eval()

    # save the model
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(os.path.join(save_dir, word2idx_filename), 'w') as f:
        json.dump(text8.word2idx, f)

    np.savez(os.path.join(save_dir, weights_filename), W_eval, V_eval)

    return text8.word2idx, W_eval, V_eval


def get_negative_sampling_distribution(indexed_corpus, vocab_size):
    print(f'* calling {get_negative_sampling_distribution.__name__}')

    # Pn(w) = prob of word occurring
    # we would like to sample the negative samples
    # such that words that occur more often should be sampled more often

    # prevent divided by 0
    word_freq = np.ones(vocab_size)
    for word in indexed_corpus:
        word_freq[word] += 1

    # smooth it
    # if we didn't add (** 0.75)
    # the infrequent words are too infrequent
    # and so they are very unlikely to ever be sampled
    p_neg = word_freq ** 0.75

    # normalization
    p_neg = p_neg / p_neg.sum()

    return p_neg


def load_model():
    with open(os.path.join(save_dir, word2idx_filename)) as f:
        word2idx = json.load(f)
    npz = np.load(os.path.join(save_dir, weights_filename))
    W = npz['arr_0']
    V = npz['arr_1']
    return word2idx, W, V


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

    if os.path.exists(os.path.join(save_dir, word2idx_filename)) and \
       os.path.exists(os.path.join(save_dir, weights_filename)):
        word2idx, W, V = load_model()
    else:
        word2idx, W, V = train_model()
    test_model(word2idx, W, V)
