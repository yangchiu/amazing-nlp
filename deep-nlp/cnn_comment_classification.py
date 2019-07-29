import numpy as np
import os
import re
import pandas as pd
from keras.preprocessing.text import Tokenizer

# GloVe: https://nlp.stanford.edu/projects/glove/
# direct download link: http://nlp.stanford.edu/data/glove.6B.zip
glove_filepath = '../pretrained-word-embeddings/glove.6B/glove.6B.100d.txt'

save_dir = 'cnn_comment_classification/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

data_filename = 'train.csv'


def load_word_vectors(filepath):
    print(f'* calling {load_word_vectors.__name__}')

    embedding = []
    word2idx = {}
    idx2word = []
    with open(glove_filepath, encoding='utf-8') as f:
        # it's just a space-separated text file in the format:
        # word vec[0] vec[1] vec[2] ...
        for i, line in enumerate(f):

            values = line.split()

            word = values[0]
            vector = np.array(values[1:], dtype=np.float32)

            embedding.append(vector)
            idx2word.append(word)
            word2idx[word] = i

        print(f'=> found {len(idx2word)} word vectors')

    embedding = np.array(embedding)
    vocab_size, embedding_size = embedding.shape

    return embedding, word2idx, idx2word, vocab_size, embedding_size


class ToxicComments:

    def __init__(self):
        print(f'* calling {ToxicComments.__init__.__name__}')

        self.comments = None
        self.targets = None
        self.possible_labels = None
        self.load_data()

        self.tokenizer = None
        self.indexed_comments = None
        self.tokenize()

    def load_data(self):
        print(f'* calling {ToxicComments.load_data.__name__}')

        train = pd.read_csv(os.path.join(save_dir, data_filename))

        self.comments = train['comment_text'].values

        # replace \xa0 with white-space
        for i, line in enumerate(self.comments):
            self.comments[i] = MultiReplacer.replace(self.comments[i])

        print(f'=> get {len(self.comments)} comments')

        self.possible_labels = train.columns.values[2:]
        print(f'=> possible labels are:')
        print(self.possible_labels)

        # one comment could have one or more labels
        self.targets = train[self.possible_labels].values

    def tokenize(self):
        print(f'* calling {ToxicComments.tokenize.__name__}')

        self.tokenizer = Tokenizer(num_words=50000)
        self.tokenizer.fit_on_texts(self.comments)
        self.indexed_comments = self.tokenizer.texts_to_sequences(self.comments)

        print(self.tokenizer.word_index)


class MultiReplacer:
    dict = {
        # &nbsp
        '\xa0': ' ',
        # dash
        '\u2011': '\u002D',
        '\u2212': '\u002D',
        '\u2013': '\u002D',
        '\u2014': '\u002D',
        # single quotation
        '\u2018': '\u0027',
        '\u2019': '\u0027',
        # left double quotation
        '\u201C': '\u0022',
        # right double quotation
        '\u201D': '\u0022'
    }

    regex = re.compile(f'{"|".join(map(re.escape, dict.keys()))}')

    @classmethod
    def replace(cls, text):
        return cls.regex.sub(
            lambda mo: cls.dict[mo.string[mo.start():mo.end()]],
            text
        )

if __name__ == '__main__':

    toxic = ToxicComments()
