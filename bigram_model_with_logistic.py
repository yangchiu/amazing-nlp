import numpy as np
import random

from brown_corpus import get_sentences_with_word2idx_limit_vocab, get_sentences_with_word2idx

if __name__ == '__main__':
    indexed_sents, word2idx = get_sentences_with_word2idx_limit_vocab(2000)

    n_vocab = len(word2idx)
    print(f'Vocab size: {n_vocab}')

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
        for sent in indexed_sents:

            sent = [start_token_idx] + sent + [end_token_idx]

            n = len(sent)

            # use (n - 1)th word to predict nth word
            # so inputs only need 0 ~ n-1 words
            # and targets only need 1 ~ n words
            inputs = np.zeros((n-1, n_vocab))
            targets = np.zeros((n-1, n_vocab))

            inputs[np.arange(n-1), sent[:n-1]] = 1
            targets[np.arange(n-1), sent[1:]] = 1

            predictions = softmax(inputs.dot(W))

            W = W - learning_rate * inputs.T.dot(predictions - targets)

            loss = -np.sum(targets * np.log(predictions)) / (n - 1)
            losses.append(loss)

            if j % 10 == 0:
                print(f'epoch: {epoch}, sentence: {j}/{len(indexed_sents)}, loss: {loss}')

            j += 1

