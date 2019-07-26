import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier

save_dir = 'bow_classifier/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

training_data_filename = 'r8-train-all-terms.txt'
test_data_filename = 'r8-test-all-terms.txt'

glove_filepath = '../pretrained-word-embeddings/glove.6B/glove.6B.50d.txt'


class GloveVectorizer:

    def __init__(self):
        print(f'* calling {GloveVectorizer.__init__.__name__}')

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

        self.embedding = embedding
        self.word2idx = word2idx
        self.idx2word = idx2word
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size

    def transform(self, sentences):
        print(f'* calling {GloveVectorizer.transform.__name__}')

        vectorized_bow = np.zeros((len(sentences), self.embedding_size))

        empty_count = 0
        for i, sent in enumerate(sentences):

            words = sent.lower().split()

            vectors_for_words = []
            for word in words:
                if word in self.word2idx:
                    vector = self.embedding[self.word2idx[word]]
                    # collect all word vectors in this sentence
                    vectors_for_words.append(vector)
                else:
                    #print(f'=> out of vocabulary word: {word} in line {i}')
                    pass

            if len(vectors_for_words) > 0:
                vectors_for_words = np.array(vectors_for_words)
                vectorized_bow[i] = vectors_for_words.mean(axis=0)
            else:
                print(f'=> empty line {i}')
                empty_count += 1

        print(f'=> {empty_count} empty lines')

        return vectorized_bow


if __name__ == '__main__':

    train = pd.read_csv(os.path.join(save_dir, training_data_filename),
                        # if there's no header (column names) in your file, set header=None
                        header=None,
                        sep='\t')
    train.columns = ['label', 'content']

    test = pd.read_csv(os.path.join(save_dir, test_data_filename),
                       header=None,
                       sep='\t')
    test.columns = ['label', 'content']

    vectorizer = GloveVectorizer()

    x_train = vectorizer.transform(train.content)
    y_train = train.label

    x_test = vectorizer.transform(test.content)
    y_test = test.label

    model = RandomForestClassifier(n_estimators=200)
    model.fit(x_train, y_train)
    print(f'=> train score: {model.score(x_train, y_train)}')
    print(f'=> test score: {model.score(x_test, y_test)}')
