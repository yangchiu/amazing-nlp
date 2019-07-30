import os
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# GloVe: https://nlp.stanford.edu/projects/glove/
# direct download link: http://nlp.stanford.edu/data/glove.6B.zip
glove_filepath = '../pretrained-word-embeddings/glove.6B/glove.6B.100d.txt'

save_dir = 'lstm_poetry_generation/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

corpus_filename = 'robert_frost.txt'


class Glove:

    def __init__(self, filepath):
        print(f'* calling Glove()')

        self.embedding = []
        self.word2idx = {}
        self.idx2word = []
        with open(filepath, encoding='utf-8') as f:
            # it's just a space-separated text file in the format:
            # word vec[0] vec[1] vec[2] ...
            for i, line in enumerate(f):

                values = line.split()

                word = values[0]
                vector = np.array(values[1:], dtype=np.float32)

                self.embedding.append(vector)
                self.idx2word.append(word)
                self.word2idx[word] = i

            print(f'=> found {len(self.idx2word)} word vectors')

        self.embedding = np.array(self.embedding)
        self.vocab_size, self.embedding_size = self.embedding.shape


class RobertFrostCorpus:

    def __init__(self, pretrained_word_embedding):

        self.pretrained = pretrained_word_embedding

        self.input_lines = []
        self.target_lines = []
        self.get_corpus()

        self.tokenizer = None
        self.vocab_size = 50000
        self.max_seq_len = 0
        self.input_sequences = None
        self.target_sequences = None
        self.tokenize()

        self.embedding = None
        self.build_word_embedding()

    def get_corpus(self):
        print(f'* calling {RobertFrostCorpus.get_corpus.__name__}')

        for line in open(os.path.join(save_dir, corpus_filename), encoding='utf8'):
            # remove trailing whitespace
            line = line.rstrip()
            # empty line
            if not line:
                continue

            # manipulate the original line to produce input and target line
            # we will use text(t) to predict text(t+1)
            # shift input by 1 step to produce target
            #
            # example:
            #
            #   t1    t2   t3  t4  t5    t6
            # <sos>  this  is  a  book  <eos>
            #
            #            t1   |   t2  |   t3  |   t4  |   t5
            # input  =  <sos> |  this |   is  |   a   |  book
            # target =  this  |   is  |   a   |  book |  <eos>
            #            t2       t3      t4      t5      t6
            input_line = '<sos> ' + line
            target_line = line + ' <eos>'

            self.input_lines.append(input_line)
            self.target_lines.append(target_line)

        print(f'=> total {len(self.input_lines)} input lines')
        print(f'=> total {len(self.target_lines)} target lines')

    def tokenize(self):
        print(f'* calling {RobertFrostCorpus.tokenize.__name__}')

        all_lines = self.input_lines + self.target_lines

        # set filters to '', else special characters would be removed,
        # and cause <sos> <eos> become sos eos
        self.tokenizer = Tokenizer(num_words=self.vocab_size,
                                   filters='')
        self.tokenizer.fit_on_texts(all_lines)
        self.input_sequences = self.tokenizer.texts_to_sequences(self.input_lines)
        self.target_sequences = self.tokenizer.texts_to_sequences(self.target_lines)

        self.max_seq_len = max(len(s) for s in self.input_sequences)
        print(f'=> max sequence length = {self.max_seq_len}')

        self.input_sequences = pad_sequences(self.input_sequences,
                                             maxlen=self.max_seq_len,
                                             padding='post')
        self.target_sequences = pad_sequences(self.target_sequences,
                                              maxlen=self.max_seq_len,
                                              padding='post')
        print(f'=> input sequences shape = {self.input_sequences.shape}')
        print(f'=> target sequences shape = {self.target_sequences.shape}')

    def build_word_embedding(self):
        print(f'* calling {RobertFrostCorpus.build_word_embedding.__name__}')

        # using pretrained word embedding to build
        # word embedding for words in robert frost corpus

        # becuase keras tokenizer reserve index 0 for <pad>
        num_words = self.vocab_size + 1

        self.embedding = np.zeros((num_words, self.pretrained.embedding_size))

        # find a word vector for every word in tokenizer.word_index if i < vocab_size
        # word vector for <pad> is all zeros
        for word, i in self.tokenizer.word_index.items():
            if i < self.vocab_size:
                index = self.pretrained.word2idx.get(word)
                # word vector for a word not found in pretrained will be all zeros
                if index is not None:
                    self.embedding[i] = self.pretrained.embedding[index]
        print(f'=> get word embedding with shape = {self.embedding.shape}')

        # one-hot target
        # because sparse categorical cross entropy
        # only works for one input -> one target
        # but now we have one input -> T targets
        # ==> do this.
        #
        # Well unfortunately if you pass in the sparse categorical cross entropy loss function into Cara's it
        #
        # won't work when each target is a sequence.
        #
        # This was OK in our eminent example because in essence you only have one target per input.
        #
        # But now we have two targets per input or in other words.
        #
        # Each sample gives us an entire sequence of targets so sparse categorical cross entropy just wasn't written
        #
        # for that case.


if __name__ == '__main__':

    glove = Glove(glove_filepath)
    corpus = RobertFrostCorpus(glove)
