from nltk.corpus import brown
import operator
import re

import nltk
nltk.download('brown')


def get_sentences():
    print(f'* calling {get_sentences.__name__}')

    sents = brown.sents()

    # only keep words, remove punctuations
    regexp = re.compile(r'\w')

    new_sents = []
    for sent in sents:
        for j, word in reversed(list(enumerate(sent))):
            if not regexp.search(word):
                del sent[j]
            else:
                sent[j] = sent[j].lower()
        new_sents.append(sent)

    return new_sents


def get_sentences_with_word2idx_limit_vocab(n_vocab=10000, min_sent_length=5):
    print(f'* calling {get_sentences_with_word2idx_limit_vocab.__name__}')

    print(f'=> min sentence length = {min_sent_length}')

    sents = get_sentences()

    # build word counts and temp word2idx
    word_count = {}
    for sent in sents:
        if len(sent) < min_sent_length:
            continue
        for token in sent:
            word_count[token] = word_count.get(token, 0) + 1

    # restrict vocab size
    word_count = sorted(word_count.items(), key=operator.itemgetter(1), reverse=True)

    # <sos> and <eos> token are necessary for glove model
    # let index start from 2, reserve spaces for <sos> and <eos>
    word2idx = {}
    i = 2
    for word, count in word_count[:n_vocab]:
        word2idx[word] = i
        i += 1
    # add an extra idx for '<unk>'
    word2idx['<unk>'] = i
    word2idx['<sos>'] = 0 # start of sentence
    word2idx['<eos>'] = 1 # end of sentence

    print(f'=> get word2idx with vocab size: {len(word2idx)}')

    indexed_sents = []
    for sent in sents:
        if len(sent) < min_sent_length:
            continue
        indexed_sent = [word2idx[word] if word in word2idx else word2idx['<unk>'] for word in sent]
        indexed_sents.append(indexed_sent)

    return indexed_sents, word2idx


def get_idx2word(word2idx):
    idx2word = dict((v, k) for k, v in word2idx.items())
    return idx2word


def get_words_from_idx(indexed_sent, idx2word):
    return [idx2word[i] for i in indexed_sent]


if __name__ == '__main__':
    sents, word2idx = get_sentences_with_word2idx_limit_vocab()

    idx2word = get_idx2word(word2idx)

    for sent in sents[:30]:
        orig_sent = get_words_from_idx(sent, idx2word)
        string = ''
        for i, word in enumerate(orig_sent):
            string += f'{word} ({sent[i]}) '
        print(f'[sample sentence] {string}')
