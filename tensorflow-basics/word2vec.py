import collections
from collections import Counter
import os
import random
import zipfile
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
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
from six.moves import xrange

save_dir = 'word2vec/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

data_dir = os.path.join(save_dir, 'data/')
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

data_url = 'http://mattmahoney.net/dc/text8.zip'

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
        labels = np.ndarray(shape=(batch_size,1), dtype=np.int32)

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
                labels[i * num_skips + j, 0] = dq[target]
            # after got # of num_skip sample,
            # move to the next window
            if self.current_corpus_index == self.corpus_len:
                dq[:] = self.indexed_corpus[:span]
                self.current_corpus_index = span
            else:
                dq.append(self.indexed_corpus[self.current_corpus_index])
                self.current_corpus_index += 1
        # shift back a little bit to avoid skipping words in the end of a batch
        self.current_corpus_index = (self.current_corpus_index + self.corpus_len - span) % self.corpus_len
        return batch, labels

if __name__ == '__main__':

    text8 = Text8(data_url, data_dir)

    # constants
    batch_size = 128
    embedding_size = 150
    # how many words to consider left and right
    # window = [ skip_window <-- target --> skip_window ]
    skip_window = 1
    # how many times to reuse an input (target word) to find a label (context word)
    num_skips = 2

    # number of negative examples to sample in nce_loss
    num_neg_sampled = 64

    learning_rate = 0.01
    vocab_size = 50000

    epochs = 10000

    # placeholders
    train_inputs = tf.placeholder(tf.int32, shape=[None])
    # the shape of train_labels for nce_loss must be rank 2
    train_labels = tf.placeholder(tf.int32, shape=[None, 1])

    # variables
    embeddings = tf.Variable(
        tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0)
    )
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)

    # loss
    nce_weights = tf.Variable(
        tf.truncated_normal(
            [vocab_size, embedding_size],
            stddev=1.0/np.sqrt(embedding_size)
        )
    )
    nce_bias = tf.Variable(tf.zeros([vocab_size]))
    # tf.nce_loss automatically draws a new sample of the negative labels
    # each time we evaluate the loss
    loss = tf.reduce_mean(
        tf.nn.nce_loss(nce_weights, nce_bias,
                       train_labels, embed,
                       num_neg_sampled, vocab_size)
    )

    # optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train = optimizer.minimize(loss)

    # normalize word embeddings
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), axis=1, keep_dims=True))
    normalized_embeddings = embeddings / norm

    # init
    init = tf.global_variables_initializer()

    # session
    saver = tf.train.Saver()
    with tf.Session() as sess:

        sess.run(init)
        average_loss = 0

        for i in range(epochs):

            batch_inputs, batch_labels = text8.next_batch(batch_size, num_skips, skip_window)

            _, loss_val = sess.run([train, loss], feed_dict={
                train_inputs: batch_inputs,
                train_labels: batch_labels
            })
            average_loss += loss_val

            if i % 1000 == 0:
                if i != 0:
                    average_loss /= 1000
                print(f'=> epoch {i}, average loss = {average_loss}')
                average_loss = 0

        final_embeddings = normalized_embeddings.eval()

        saver.save(sess, f'{save_dir}/')

    def plot_with_labels(low_dim_embeddings, labels):
        assert low_dim_embeddings.shape[0] >= len(labels)
        plt.figure(figsize=(18, 18))
        for i, label in enumerate(labels):
            x, y = low_dim_embeddings[i, :]
            plt.scatter(x, y)
            plt.annotate(label,
                         xy=(x, y),
                         xytext=(5, 2),
                         textcoords='offset points',
                         ha='right',
                         va='bottom')
        plt.show()
        plt.savefig(f'{save_dir}/tsne.png')

    print(f'=> get embeddings with shape {final_embeddings.shape}')

    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    plot_only = 2000
    low_dim_embeddings = tsne.fit_transform(final_embeddings[:plot_only, :])
    labels = [text8.vocab[i] for i in range(plot_only)]
    print(f'=> get low dimension embeddings with shape {low_dim_embeddings.shape}')

    plot_with_labels(low_dim_embeddings, labels)

    np.save(f'{save_dir}/trained_embeddings_200k_steps', final_embeddings)
