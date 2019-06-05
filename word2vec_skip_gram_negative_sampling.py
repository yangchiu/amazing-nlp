import numpy as np
from datetime import datetime
from scipy.special import expit as sigmoid
from sklearn.metrics.pairwise import pairwise_distances
from scipy.spatial.distance import cosine as cos_dist

from brown_corpus import get_sentences_with_word2idx_limit_vocab, get_idx2word, get_words_from_idx

def train_model(savedir):
    print(f'* calling {train_model.__name__}')

    # get word2idx data
    indexed_sents, word2idx = get_sentences_with_word2idx_limit_vocab()
    idx2word = get_idx2word(word2idx)

    # number of unique words
    vocab_size = len(word2idx)
    print(f'=> build word2vec with vocab size: {vocab_size}')

    # config
    window_size = 5
    learning_rate = 0.025
    final_learning_rate = 0.0001
    num_negatives = 5 # number of negative samples to draw per input word
    epochs = 20
    D = 50 # word embedding size

    print(f'=> word2vec dimension is {D}')

    # learning rate decay
    learning_rate_delta = (learning_rate - final_learning_rate) / epochs

    # weight matrix
    W = np.random.randn(vocab_size, D) # input-to-hidden
    V = np.random.randn(D, vocab_size) # hidden-to-output

    print(f'=> W is {W.shape}')
    print(f'=> V is {V.shape}')

    # distribution for drawing negative samples
    p_neg = get_negative_sampling_distribution(indexed_sents, vocab_size)

    # save the costs
    costs = []

    # number of total words in corpus
    #total_words = sum(len(sent) for sent in indexed_sents)
    #print(f'=> total number of words in corpus: {total_words}')

    # common words are very common and uncommon words are very uncommon
    # that means we'd spend a majority of the time updating word of vectors for very common words
    # so each time we encounter a sentence we randomly drop some words according to some probability distribution
    # if p(w) = 1e-5, then p_drop(w) = 1 - 1 = 0
    # if p(w) = 0.1, then p_drop(w) = 1 - 1e-2 = 0.99
    threshold = 1e-5
    p_drop = 1 - np.sqrt(threshold / p_neg)

    for epoch in range(epochs):
        np.random.shuffle(indexed_sents)

        cost = 0
        t0 = datetime.now()
        for i, sent in enumerate(indexed_sents):
            # drop words from this sentence
            # drop words based on p_neg
            sent = [w for w in sent if np.random.random() < (1 - p_drop[w])]
            if len(sent) < 5:
                continue

            print(f'=> sentence before drop: {get_words_from_idx(indexed_sents[i], idx2word)}')
            print(f'=> sentence after drop: {get_words_from_idx(sent, idx2word)}')

            print(f'=> sentence length: {len(sent)}')

            # randomize positions for this sentence
            randomly_ordered_positions = np.random.choice(
                len(sent),
                size=len(sent),
                replace=False
            )
            print(f'=> randomized sentence positions: {randomly_ordered_positions}')

            # train on this sentence
            for pos in randomly_ordered_positions:
                # the middle word
                word = sent[pos]
                print(f'=> select word "{get_words_from_idx([word], idx2word)}"')

                context_words = get_context(pos, sent, window_size)
                print(f'=> get context {get_words_from_idx(context_words, idx2word)}')

                # usually negative sampling is sampling negative context words
                # with a fixed middle word
                # but during implementation we fix the context words
                # and insert an incorrect middle word
                neg_word = np.random.choice(vocab_size, p=p_neg)
                print(f'=> get neg_word: {get_words_from_idx([neg_word], idx2word)}')

                targets = np.array(context_words)
                c = sgd(word, targets, 1, learning_rate, W, V)
                cost += c
                c = sgd(neg_word, targets, 0, learning_rate, W, V)
                cost += c

            if i % 10 == 0:
                print(f'********************** processed {i} / {len(indexed_sents)} **********************')

        dt = datetime.now() - t0
        print(f'epoch complete: {epoch}, cost: {cost}, dt: {dt}')

        costs.append(cost)

        learning_rate -= learning_rate_delta

    return word2idx, W, V


def sgd(input, targets, label, learning_rate, W, V):
    #print(f'* calling {sgd.__name__}')

    # W[input] shape: D
    # V[:,targets] shape: D x N
    # activation shape: N
    activation = W[input].dot(V[:,targets])
    prob = sigmoid(activation)

    gV = np.outer(W[input], prob - label) # D x N
    gW = np.sum((prob - label) * V[:,targets], axis=1) # D

    V[:,targets] -= learning_rate * gV
    W[input] -= learning_rate * gW

    cost = label * np.log(prob + 1e-10) + (1 - label) * np.log(1 - prob + 1e-10)
    return cost.sum()


def get_context(pos, sentence, window_size):
    print(f'* calling {get_context.__name__}')

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


def get_negative_sampling_distribution(indexed_sents, vocab_size):
    print(f'* calling {get_negative_sampling_distribution.__name__}')

    # Pn(w) = prob of word occuring
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

    print(f'=> p_neg shape {p_neg.shape}')
    print(f'=> p_neg: {p_neg}')
    return p_neg


def analogy(pos1, neg1, pos2, neg2, word2idx, idx2word, W):
    V, D = W.shape

    # don't actually use pos2 in calculation, just print what's expected
    print("testing: %s - %s = %s - %s" % (pos1, neg1, pos2, neg2))
    for w in (pos1, neg1, pos2, neg2):
        if w not in word2idx:
            print("Sorry, %s not in word2idx" % w)
            return

    p1 = W[word2idx[pos1]]
    n1 = W[word2idx[neg1]]
    p2 = W[word2idx[pos2]]
    n2 = W[word2idx[neg2]]

    vec = p1 - n1 + n2

    distances = pairwise_distances(vec.reshape(1, D), W, metric='cosine').reshape(V)
    idx = distances.argsort()[:10]

    # pick one that's not p1, n1, or n2
    best_idx = -1
    keep_out = [word2idx[w] for w in (pos1, neg1, neg2)]
    # print("keep_out:", keep_out)
    for i in idx:
        if i not in keep_out:
            best_idx = i
            break
    # print("best_idx:", best_idx)

    print("got: %s - %s = %s - %s" % (pos1, neg1, idx2word[best_idx], neg2))
    print("closest 10:")
    for i in idx:
        print(idx2word[i], distances[i])

    print("dist to %s:" % pos2, cos_dist(p2, vec))


def test_model(word2idx, W, V):
    # there are multiple ways to get the "final" word embedding
    # We = (W + V.T) / 2
    # We = W

    idx2word = {i:w for w, i in word2idx.items()}

    for We in (W, (W + V.T) / 2):
        print("**********")

        analogy('king', 'man', 'queen', 'woman', word2idx, idx2word, We)
        analogy('king', 'prince', 'queen', 'princess', word2idx, idx2word, We)
        analogy('miami', 'florida', 'dallas', 'texas', word2idx, idx2word, We)
        analogy('einstein', 'scientist', 'picasso', 'painter', word2idx, idx2word, We)
        analogy('japan', 'sushi', 'germany', 'bratwurst', word2idx, idx2word, We)
        analogy('man', 'woman', 'he', 'she', word2idx, idx2word, We)
        analogy('man', 'woman', 'uncle', 'aunt', word2idx, idx2word, We)
        analogy('man', 'woman', 'brother', 'sister', word2idx, idx2word, We)
        analogy('man', 'woman', 'husband', 'wife', word2idx, idx2word, We)
        analogy('man', 'woman', 'actor', 'actress', word2idx, idx2word, We)
        analogy('man', 'woman', 'father', 'mother', word2idx, idx2word, We)
        analogy('heir', 'heiress', 'prince', 'princess', word2idx, idx2word, We)
        analogy('nephew', 'niece', 'uncle', 'aunt', word2idx, idx2word, We)
        analogy('france', 'paris', 'japan', 'tokyo', word2idx, idx2word, We)
        analogy('france', 'paris', 'china', 'beijing', word2idx, idx2word, We)
        analogy('february', 'january', 'december', 'november', word2idx, idx2word, We)
        analogy('france', 'paris', 'germany', 'berlin', word2idx, idx2word, We)
        analogy('week', 'day', 'year', 'month', word2idx, idx2word, We)
        analogy('week', 'day', 'hour', 'minute', word2idx, idx2word, We)
        analogy('france', 'paris', 'italy', 'rome', word2idx, idx2word, We)
        analogy('paris', 'france', 'rome', 'italy', word2idx, idx2word, We)
        analogy('france', 'french', 'england', 'english', word2idx, idx2word, We)
        analogy('japan', 'japanese', 'china', 'chinese', word2idx, idx2word, We)
        analogy('china', 'chinese', 'america', 'american', word2idx, idx2word, We)
        analogy('japan', 'japanese', 'italy', 'italian', word2idx, idx2word, We)
        analogy('japan', 'japanese', 'australia', 'australian', word2idx, idx2word, We)
        analogy('walk', 'walking', 'swim', 'swimming', word2idx, idx2word, We)

if __name__ == '__main__':
    word2idx, W, V = train_model('w2v_model')
    test_model(word2idx, W, V)