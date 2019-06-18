import os
import numpy as np
import tensorflow as tf
import json
from pathlib import Path
from datetime import datetime

from sklearn.metrics.pairwise import pairwise_distances
from scipy.spatial.distance import cosine as cos_dist

from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')

# tensorflow-gpu version log:
# windows 10
# cuda 10.0
# cudnn 7.4.1.5
# tensorflow-gpu 1.13.1

from opinrank_corpus import get_sentences_with_word2idx_limit_vocab, get_idx2word, get_words_from_idx

savedir = 'trained_models/word2vec_tf'

def train_model():
    print(f'* calling {train_model.__name__}')

    # get word2idx data
    indexed_sents, word2idx = get_sentences_with_word2idx_limit_vocab()
    idx2word = get_idx2word(word2idx)

    # number of unique words
    vocab_size = len(word2idx)
    print(f'=> build word2vec with vocab size: {vocab_size}')

    # config
    window_size = 10
    epochs = 20
    D = 50  # word embedding size

    print(f'=> word2vec dimension is {D}')

    # distribution for drawing negative samples
    p_neg = get_negative_sampling_distribution(indexed_sents, vocab_size)
    print(f'=> p_neg shape {p_neg.shape}')
    print(f'=> p_neg: {p_neg}')
    print(f'=> p_neg max: {p_neg.max()}')
    print(f'=> p_neg min: {p_neg.min()}')
    print(f'=> p_neg for [the] = {p_neg[word2idx["the"]]}')
    print(f'=> p_neg for [by] = {p_neg[word2idx["by"]]}')
    print(f'=> p_neg for [this] = {p_neg[word2idx["this"]]}')
    print(f'=> p_neg for [animal] = {p_neg[word2idx["animal"]]}')
    print(f'=> p_neg for [scale] = {p_neg[word2idx["scale"]]}')

    # common words are very common and uncommon words are very uncommon
    # that means we'd spend a majority of the time updating word of vectors for very common words
    # so each time we encounter a sentence we randomly drop some words according to some probability distribution
    # if threshold = 1e-5 and p(w) = 1e-5, then p_drop(w) = 1 - 1 = 0
    # if threshold = 1e-5 and p(w) = 0.1, then p_drop(w) = 1 - 1e-2 = 0.99
    # *** Adjust this value according to p_neg ***
    # *** Until common words can be dropped by a relative larger pos ***
    threshold = 4e-5
    p_drop = 1 - np.sqrt(threshold / p_neg)

    stop_words = set(stopwords.words('english'))
    index = [word2idx[word] if word in word2idx else word2idx['<UNK>'] for word in stop_words]
    p_drop[index] = 0.9

    print(f'=> p_drop shape {p_drop.shape}')
    print(f'=> p_drop: {p_drop}')
    print(f'=> p_drop max: {p_drop.max()}')
    print(f'=> p_drop min: {p_drop.min()}')
    print(f'=> p_drop for [the] = {p_drop[word2idx["the"]]}')
    print(f'=> p_drop for [by] = {p_drop[word2idx["by"]]}')
    print(f'=> p_drop for [this] = {p_drop[word2idx["this"]]}')
    print(f'=> p_drop for [animal] = {p_drop[word2idx["animal"]]}')
    print(f'=> p_drop for [scale] = {p_drop[word2idx["scale"]]}')

    # weight matrix
    W = np.random.randn(vocab_size, D).astype(np.float32)  # input-to-hidden
    V = np.random.randn(D, vocab_size).astype(np.float32)  # hidden-to-output
    print(f'=> W is {W.shape}')
    print(f'=> V is {V.shape}')

    # create model
    tf_posword = tf.placeholder(tf.int32, shape=(None,))
    tf_negword = tf.placeholder(tf.int32, shape=(None,))
    tf_context = tf.placeholder(tf.int32, shape=(None,))
    tf_W = tf.Variable(W)
    tf_V = tf.Variable(V.T)

    def dot(A, B):
        C = A * B
        return tf.reduce_sum(C, axis=1)

    # correct middle word output (posword + context)
    # embedding_lookup is used to look up indexes in a tensor
    emb_input = tf.nn.embedding_lookup(tf_W, tf_posword) # 1 x D
    emb_output = tf.nn.embedding_lookup(tf_V, tf_context) # N x D
    correct_output = dot(emb_input, emb_output) # N
    pos_loss = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.ones(tf.shape(correct_output)),
        logits=correct_output
    )

    # incorrect middle word output (negword + context)
    emb_input = tf.nn.embedding_lookup(tf_W, tf_negword)
    incorrect_output = dot(emb_input, emb_output)
    neg_loss = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.zeros(tf.shape(incorrect_output)),
        logits=incorrect_output
    )

    loss = (tf.reduce_mean(pos_loss) + tf.reduce_mean(neg_loss)) / 2

    train_op = tf.train.MomentumOptimizer(0.1, momentum=0.9).minimize(loss)

    session = tf.Session()
    init_op = tf.global_variables_initializer()
    session.run(init_op)

    costs = []

    for epoch in range(epochs):
        np.random.shuffle(indexed_sents)

        cost = 0
        counter = 0
        poswords = []
        targets = []
        negwords = []

        t0 = datetime.now()
        for i, sent in enumerate(indexed_sents):
            # drop words from this sentence
            # drop words based on p_neg
            sent = [w for w in sent if np.random.random() < (1 - p_drop[w])]
            if len(sent) < 5:
                continue

            #print(f'****** sentence {i} ******')
            #print(f'=> sentence before drop: {get_words_from_idx(indexed_sents[i], idx2word)}')
            #print(f'=> sentence after drop: {get_words_from_idx(sent, idx2word)}')

            # print(f'=> sentence length: {len(sent)}')

            # randomize positions for this sentence
            randomly_ordered_positions = np.random.choice(
                len(sent),
                size=len(sent),
                replace=False
            )
            #print(f'=> randomized sentence positions: {randomly_ordered_positions}')

            # train on this sentence
            for j, pos in enumerate(randomly_ordered_positions):
                # the middle word
                word = sent[pos]
                # print(f'=> select word "{get_words_from_idx([word], idx2word)}"')

                context_words = get_context(pos, sent, window_size)
                # print(f'=> get context {get_words_from_idx(context_words, idx2word)}')

                # usually negative sampling is sampling negative context words
                # with a fixed middle word
                # but during implementation we fix the context words
                # and insert an incorrect middle word
                neg_word = np.random.choice(vocab_size, p=p_neg)
                # print(f'=> get neg_word: {get_words_from_idx([neg_word], idx2word)}')

                n = len(context_words)
                poswords += [word] * n
                negwords += [neg_word] * n
                targets += context_words

            if len(poswords) >= 128:
                _, c = session.run(
                    (train_op, loss),
                    feed_dict={
                        tf_posword: poswords,
                        tf_negword: negwords,
                        tf_context: targets,
                    }
                )
                cost += c

                # reset
                poswords = []
                negwords = []
                targets = []

            counter += 1
            if counter % 1000 == 0:
                print(f'********************** processed {i} / {len(indexed_sents)} **********************')

        dt = datetime.now() - t0
        print(f'epoch complete: {epoch}, cost: {cost / len(indexed_sents)}, dt: {dt.seconds // 60} mins {dt.seconds % 60} secs')

        costs.append(cost)

    # save the model
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    with open(f'{savedir}/word2idx.json', 'w') as f:
        json.dump(word2idx, f)

    np.savez(f'{savedir}/weights.npz', W, V)

    return word2idx, W, V


def get_negative_sampling_distribution(indexed_sents, vocab_size):
    print(f'* calling {get_negative_sampling_distribution.__name__}')

    # Pn(w) = prob of word occurring
    # we would like to sample the negative samples
    # such that words that occur more often should be sampled more often

    # prevent divided by 0
    word_freq = np.ones(vocab_size)
    word_count = sum(len(sent) for sent in indexed_sents)
    print(f'=> word count for neg sampling distribution: {word_count}')
    for sent in indexed_sents:
        for word in sent:
            word_freq[word] += 1

    # smooth it
    # if we didn't add (** 0.75)
    # the infrequent words are too infrequent
    # and so they are very unlikely to ever be sampled
    p_neg = word_freq ** 0.75

    # normalization
    p_neg = p_neg / p_neg.sum()

    return p_neg


def get_context(pos, sentence, window_size):
    #print(f'* calling {get_context.__name__}')

    # input:
    # a sentence of the form: x x x x c c c pos c c c x x x x
    # output:
    # the context word indices: c c c c c c
    start = max(0, pos - window_size)
    end = min(len(sentence), pos + window_size)

    context = []
    for ctx_pos, ctx_word_idx in enumerate(sentence[start:end], start=start):
        if ctx_pos != pos:
            # exclude the input word itself
            context.append(ctx_word_idx)
    return context


def load_model():
    with open(f'{savedir}/word2idx.json') as f:
        word2idx = json.load(f)
    npz = np.load(f'{savedir}/weights.npz')
    W = npz['arr_0']
    V = npz['arr_1']
    return word2idx, W, V


def test_model(word2idx, W, V):
    print(f'* calling {test_model.__name__}')

    idx2word = get_idx2word(word2idx)

    # there are multiple ways to get the "final" word embedding
    # We = (W + V.T) / 2
    # We = W

    # use We = W here
    # We = (W + V.T) / 2
    We = W
    print(f'****** Model Testing Start ******')

    similar('dirty', word2idx, idx2word, We)
    similar('polite', word2idx, idx2word, We)
    similar('france', word2idx, idx2word, We)
    similar('shocked', word2idx, idx2word, We)
    similar('bed', word2idx, idx2word, We)
    similar('couch', word2idx, idx2word, We)
    similar('smelly', word2idx, idx2word, We)
    similar('clean', word2idx, idx2word, We)
    similar('dog', word2idx, idx2word, We)


def similar(word, word2idx, idx2word, W):

    V, D = W.shape

    print(f'*** testing: {word} ***')
    if word not in word2idx:
        print(f'Sorry, {w} not in word2idx')
        return

    vec = W[word2idx[word]]

    distances = pairwise_distances(vec.reshape(1, D), W, metric='cosine').reshape(V)
    idx = distances.argsort()[:6]

    print(f'*** closest 6 ***')
    for i in idx:
        print(idx2word[i], distances[i])


if __name__ == '__main__':
    if Path(f'{savedir}/word2idx.json').is_file() and Path(f'{savedir}/weights.npz').is_file():
        word2idx, W, V = load_model()
    else:
        word2idx, W, V = train_model()
    test_model(word2idx, W, V)
