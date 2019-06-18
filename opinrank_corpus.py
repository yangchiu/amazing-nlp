import gzip
import gensim
import operator

data_file = './training_data/opinrank.txt.gz'


def get_sentences():
    print(f'* calling {get_sentences.__name__}')

    sents = []
    with gzip.open(data_file, 'rb') as f:
        for i, line in enumerate(f):
            try:
                sent = line.decode('utf-8').split('\t')[2]
                sent = gensim.utils.simple_preprocess(sent)
                sents.append(sent)
            except:
                pass
    return sents


def get_sentences_with_word2idx_limit_vocab(n_vocab=20000, min_sent_length=9):
    print(f'* calling {get_sentences_with_word2idx_limit_vocab.__name__}')

    sents = get_sentences()
    indexed_sents = []

    i = 4
    word2idx = {
        '<PAD>': 0,
        '<START>': 1,
        '<END>': 2,
        '<UNK>': 3
    }
    idx2word = ['<PAD>', '<START>', '<END>', '<UNK>']

    word_idx_count = {
        0: float('inf'),
        1: float('inf'),
        2: float('inf'),
        3: float('inf')
    }

    for sent in sents:
        if len(sent) < min_sent_length:
            continue
        indexed_sent = []
        for token in sent:
            token = token.lower()
            if token not in word2idx:
                idx2word.append(token)
                word2idx[token] = i
                i += 1

            idx = word2idx[token]
            # if there's already idx in word_idx_count, increment it by 1
            # else set the idx to 0 + 1
            word_idx_count[idx] = word_idx_count.get(idx, 0) + 1

            indexed_sent.append(idx)
        indexed_sents.append(indexed_sent)

    print(f'=> min sentence length is {min_sent_length}')
    print(f'=> get {len(indexed_sents)} indexed sentences')

    # restrict vocab size
    word_idx_count = sorted(word_idx_count.items(), key=operator.itemgetter(1), reverse=True)
    new_word2idx = {}
    new_idx = 0
    idx2newidx = {}
    for idx, count in word_idx_count[:n_vocab]:
        word = idx2word[idx]
        new_word2idx[word] = new_idx
        idx2newidx[idx] = new_idx
        new_idx += 1

    print(f'=> get word2idx with vocab size {len(new_word2idx)}')

    # map old idx to new idx
    new_indexed_sents = []
    for sent in indexed_sents:
        new_indexed_sent = [idx2newidx[idx] if idx in idx2newidx else new_word2idx['<UNK>'] for idx in sent]
        new_indexed_sents.append(new_indexed_sent)

    return new_indexed_sents, new_word2idx


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
