import numpy as np
import random

from brown_corpus import get_sentences_with_word2idx_limit_vocab, get_idx2word, get_words_from_idx

if __name__ == '__main__':

    indexed_sents, word2idx = get_sentences_with_word2idx_limit_vocab(2000)
    idx2word = get_idx2word(word2idx)

    n_vocab = len(word2idx)
    print(f'=> build bigram model with vocab size: {n_vocab}')

    start_token_idx = word2idx['START']
    end_token_idx = word2idx['END']

    # to avoid very large weight
    # maybe we can restrict the random values in other ways
    W = np.random.randn(n_vocab, n_vocab) / np.sqrt(n_vocab)

    losses = []
    # logistic regression runs very slow
    # so run it only 1 epoch
    epochs = 1
    learning_rate = 1e-1

    def softmax(a):
        # minus max to improve the numerical stability
        # which avoid exp function output extreme large numbers
        a = a - a.max()
        return np.exp(a) / np.exp(a).sum(axis=1, keepdims=True)

    for epoch in range(epochs):
        random.shuffle(indexed_sents)

        j = 0
        # training W[N x N] is too slow
        # train a small dataset only
        for sent in indexed_sents[:250]:

            sent = [start_token_idx] + sent + [end_token_idx]

            n = len(sent)

            # use (n - 1)th word to predict nth word
            # so inputs only need 0 ~ n-1 words
            # and targets only need 1 ~ n words
            inputs = np.zeros((n-1, n_vocab))
            targets = np.zeros((n-1, n_vocab))

            # one hot encoding
            inputs[np.arange(n-1), sent[:n-1]] = 1
            targets[np.arange(n-1), sent[1:]] = 1

            predictions = softmax(inputs.dot(W))

            W = W - learning_rate * inputs.T.dot(predictions - targets)

            loss = -np.sum(targets * np.log(predictions)) / (n - 1)
            losses.append(loss)

            if j % 10 == 0:
                print(f'epoch: {epoch}, sentence: {j}/{len(indexed_sents)}, loss: {loss}')

            j += 1

    def get_score(sent):
        score = 0
        for i in range(len(sent)):
            # get prob of start token -> the 1st word
            if i == 0:
                score += W[start_token_idx, sent[i]]
            else:
                score += W[sent[i - 1], sent[i]]
        # get prob of the last word -> end token
        score += W[sent[-1], end_token_idx]

        # normalize the log score
        return round(score / (len(sent) + 1), 2)

    # generate an uniform distribution fake bigram_probs
    # to sample a fake sentence
    # to compare the log scores of real sentence and fake sentence
    fake_bigram_probs = np.ones(n_vocab)
    fake_bigram_probs[start_token_idx] = 0
    fake_bigram_probs[end_token_idx] = 0
    fake_bigram_probs /= fake_bigram_probs.sum()

    while True:
        # get a random real sentence
        rand = np.random.choice(len(indexed_sents))
        real_sent = indexed_sents[rand]

        # get a random fake sentence
        fake_sent = np.random.choice(n_vocab, size=len(real_sent), p=fake_bigram_probs)

        # the score of real sentence would always higher than the score of fake sentence
        print(f'[Real sentence]')
        print(f'{" ".join(get_words_from_idx(real_sent, idx2word))} => Score: {get_score(real_sent)}')
        print(f'[Fake sentence]')
        print(f'{" ".join(get_words_from_idx(fake_sent, idx2word))} => Score: {get_score(fake_sent)}')

        cont = input("Continue? [Y/n]")
        if cont and cont.lower() in ('N', 'n'):
            break