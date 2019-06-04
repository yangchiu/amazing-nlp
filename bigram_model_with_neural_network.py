import numpy as np
import random

from brown_corpus import get_sentences_with_word2idx_limit_vocab, get_sentences_with_word2idx

if __name__ == '__main__':
    indexed_sents, word2idx = get_sentences_with_word2idx_limit_vocab(2000)

    n_vocab = len(word2idx)
    print(f'Vocab size: {n_vocab}')

    start_token_idx = word2idx['START']
    end_token_idx = word2idx['END']

    n_hidden = 100
    w_ih = np.random.randn(n_vocab, n_hidden) / np.sqrt(n_vocab)
    w_ho = np.random.randn(n_hidden, n_vocab) / np.sqrt(n_hidden)

    losses = []
    epochs = 1
    learning_rate = 1e-2

    def softmax(a):
        a = a - a.max()
        return np.exp(a) / np.exp(a).sum(axis=1, keepdims=True)

    for epoch in range(epochs):

        random.shuffle(indexed_sents)

        j = 0
        for sent in indexed_sents:

            sent = [start_token_idx] + sent + [end_token_idx]

            n = len(sent)

            inputs = np.zeros((n-1, n_vocab))
            targets = np.zeros((n-1, n_vocab))

            # one hot encoding
            inputs[np.arange(n-1), sent[:n-1]] = 1
            targets[np.arange(n-1), sent[1:]] = 1

            # make predictions
            hidden = np.tanh(inputs.dot(w_ih))
            predictions = softmax(hidden.dot(w_ho))

            # gradient descent
            w_ho = w_ho - learning_rate * hidden.T.dot(predictions - targets)
            diff_hidden = (predictions - targets).dot(w_ho.T) * (1 - hidden * hidden)
            w_ih = w_ih - learning_rate * inputs.T.dot(diff_hidden)

            loss = -np.sum(targets * np.log(predictions)) / (n - 1)
            losses.append(loss)

            if j % 10 == 0:
                print(f'epoch: {epoch}, sentence: {j}/{len(indexed_sents)}, loss: {loss}')

            j += 1
