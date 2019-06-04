import numpy as np
from datetime import datetime
from scipy.special import expit as sigmoid

from brown_corpus import get_sentences_with_word2idx_limit_vocab, get_sentences_with_word2idx

from glob import glob

import os
import sys
import string

def remove_punctuation(s):
    return s.translate(str.maketrans('','',string.punctuation))

def get_wiki():
  V = 20000
  files = glob('../data/enwiki-xml.txt')
  all_word_counts = {}
  for f in files:
    for line in open(f, encoding="utf-8"):
      if line and line[0] not in '[*-|=\{\}':
        s = remove_punctuation(line).lower().split()
        if len(s) > 1:
          for word in s:
            if word not in all_word_counts:
              all_word_counts[word] = 0
            all_word_counts[word] += 1
  print("finished counting")

  V = min(V, len(all_word_counts))
  all_word_counts = sorted(all_word_counts.items(), key=lambda x: x[1], reverse=True)

  top_words = [w for w, count in all_word_counts[:V-1]] + ['<UNK>']
  word2idx = {w:i for i, w in enumerate(top_words)}
  unk = word2idx['<UNK>']

  sents = []
  for f in files:
    for line in open(f, encoding="utf-8"):
      if line and line[0] not in '[*-|=\{\}':
        s = remove_punctuation(line).lower().split()
        if len(s) > 1:
          # if a word is not nearby another word, there won't be any context!
          # and hence nothing to train!
          sent = [word2idx[w] if w in word2idx else unk for w in s]
          sents.append(sent)
  return sents, word2idx

def train_model(savedir):
    # get word2idx data
    indexed_sents, word2idx = get_wiki()

    # number of unique words
    vocab_size = len(word2idx)
    print(f'Vocab size: {vocab_size}')

    # config
    window_size = 5
    learning_rate = 0.025
    final_learning_rate = 0.0001
    num_negatives = 5 # number of negative samples to draw per input word
    epochs = 20
    D = 50 # word embedding size

    # learning rate decay
    learning_rate_delta = (learning_rate - final_learning_rate) / epochs

    # weight matrix
    W = np.random.randn(vocab_size, D) # input-to-hidden
    V = np.random.randn(D, vocab_size) # hidden-to-output

    # distribution for drawing negative samples
    p_neg = get_negative_sampling_distribution(indexed_sents, vocab_size)

    # save the costs
    costs = []

    # number of total words in corpus
    total_words = sum(len(sent) for sent in indexed_sents)
    print(f'total number of words in corpus: {total_words}')

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
        counter = 0
        t0 = datetime.now()
        for sent in indexed_sents:
            # drop words based on p_neg
            sent = [w for w in sent if np.random.random() < (1 - p_drop[w])]
            if len(sent) < 2:
                continue

        print(f'sent len: {len(sent)}')

        randomly_ordered_positions = np.random.choice(
            len(sent),
            size=len(sent),
            replace=False
        )
        print(f'random positions: {randomly_ordered_positions}')

        for pos in randomly_ordered_positions:
            # the middle word
            word = sent[pos]
            print(f'word: {word}')

            context_words = get_context(pos, sent, window_size)
            if len(context_words) < 1:
                continue

            # usually negative sampling is sampling negative context words
            # with a fixed middle word
            # but during implementation we fix the context words
            # and insert an incorrect middle word
            neg_word = np.random.choice(vocab_size, p=p_neg)
            print(f'neg_word: {neg_word}')

            targets = np.array(context_words)
            c = sgd(word, targets, 1, learning_rate, W, V)
            cost += c
            c = sgd(neg_word, targets, 0, learning_rate, W, V)
            cost += c

        counter += 1
        if counter % 10 == 0:
            print(f'processed {counter} / {len(indexed_sents)}')

        dt = datetime.now() - t0
        print(f'epoch complete: {epoch}, cost: {cost}, dt: {dt}')

        costs.append(cost)

        learning_rate -= learning_rate_delta

    return word2idx, W, V

def sgd(input, targets, label, learning_rate, W, V):
    print(f'input shape:')
    print(input)
    print(f'targets shape: {targets.shape}')
    print(targets)
    print(f'W shape: {W.shape}')
    print(f'V shape: {V.shape}')
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
    # Pn(w) = prob of word occuring
    # we would like to sample the negative samples
    # such that words that occur more often should be sampled more often

    # prevent divided by 0
    word_freq = np.ones(vocab_size)
    word_count = sum(len(sent) for sent in indexed_sents)
    print(f'word count for neg sampling distribution: {word_count}')
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

    print(f'p_neg shape: {p_neg.shape}')
    print(p_neg)
    return p_neg

if __name__ == '__main__':
    word2idx, W, V = train_model('w2v_model')
