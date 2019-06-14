from gensim.corpora import WikiCorpus
from pathlib import Path
import time

# dump wiki data from https://dumps.wikimedia.org/enwiki/20190601/
# or other version from https://dumps.wikimedia.org/enwiki/
#wiki_dump_file = './large_files/enwiki-20190601-pages-articles-multistream1.xml-p10p30302.bz2'
wiki_dump_file = './large_files/enwiki-20190601-pages-articles-multistream.xml.bz2'
# clean corpus data would be kept here:
output_corpus_file = './large_files/enwiki.txt'

# if you encounter "UnicodeEncodeError: 'cp950' codec can't encode character"
# it means you didn't open a file with encoding='utf-8'
# or your console (terminal) isn't using utf-8 encoding
# for Pycharm:
# 1. On the Help menu, click Edit Custom VM Options.
# 2. Add -Dconsole.encoding=UTF-8
# 3. Restart PyCharm.

def make_corpus(in_file, out_file):
    print(f'* calling {make_corpus.__name__}')

    output = open(out_file, 'w', encoding='utf-8')
    print(f'=> open {out_file} for write')

    # would take hours
    t1 = time.time()
    print(f'=> make corpus based on {in_file} ...')
    print(f'=> please wait ...')
    wiki = WikiCorpus(in_file)
    t2 = time.time()
    print(f'=> processing corpus time: {t2 - t1}')

    t1 = time.time()
    print(f'=> start to write file ...')
    # extract "text" parts, and write them into lines
    i = 0
    for text in wiki.get_texts():
        output.write(bytes(' '.join(text), 'utf-8').decode('utf-8') + '\n')
        i += 1
        if i % 10000 == 0:
            print(f'Wrote {i} articles')
    output.close()
    t2 = time.time()
    print(f'=> Writing complete!')
    print(f'=> write file time: {t2 - t1}')

# total 466000 articles in this corpus
def get_sentences_with_word2idx_limit_vocab(n_vocab=20000, n_article=233000):
    print(f'* calling {get_sentences_with_word2idx_limit_vocab.__name__}')

    V = n_vocab
    all_word_counts = {}
    all_article_count = 0

    for line in open(output_corpus_file, encoding='utf-8'):
        if all_article_count > n_article:
            break
        words = line.split()
        if len(words) > 1:
            for word in words:
                if word not in all_word_counts:
                    all_word_counts[word] = 0
                all_word_counts[word] += 1
            all_article_count += 1
    print(f'=> {len(all_word_counts)} words in this corpus')
    print(f'=> {all_article_count} articles in this corpus')

    V = min(V, len(all_word_counts))

    all_word_counts = sorted(all_word_counts.items(), key=lambda x: x[1], reverse=True)

    top_words = [w for w, count in all_word_counts[:V - 1]] + ['<UNK>']

    word2idx = {w: i for i, w in enumerate(top_words)}

    unk = word2idx['<UNK>']

    indexed_sents = []
    i = 0
    for line in open(output_corpus_file, encoding='utf-8'):
        if i > n_article:
            break
        words = line.split()
        if len(words) > 1:
            # if a word is not nearby another word, there won't be any context!
            # and hence nothing to train!
            indexed_sent = [word2idx[word] if word in word2idx else unk for word in words]
            indexed_sents.append(indexed_sent)
            i += 1
    return indexed_sents, word2idx


def get_idx2word(word2idx):
    idx2word = dict((v, k) for k, v in word2idx.items())
    return idx2word


def get_words_from_idx(indexed_sent, idx2word):
    return [idx2word[i] for i in indexed_sent]


if __name__ == '__main__':
    if Path(output_corpus_file).is_file():
        print(f'=> corpus already exists!')
    else:
        make_corpus(wiki_dump_file, output_corpus_file)

    with open(output_corpus_file, encoding='utf-8') as f:
        head_lines = [next(f) for x in range(5)]
        for line in head_lines:
            print('======================')
            print(line)

    sents, word2idx = get_sentences_with_word2idx_limit_vocab()

    idx2word = get_idx2word(word2idx)

    for sent in sents[:3]:
        orig_sent = get_words_from_idx(sent, idx2word)
        string = ''
        for i, word in enumerate(orig_sent):
            string += f'{word} ({sent[i]}) '
        print(f'[sample sentence] {string}')
