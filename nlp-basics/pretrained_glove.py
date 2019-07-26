import numpy as np
from sklearn.metrics.pairwise import pairwise_distances

# gensim cannot load glove word vectors
# so load it into numpy array

# GloVe: https://nlp.stanford.edu/projects/glove/
# direct download link: http://nlp.stanford.edu/data/glove.6B.zip
glove_filepath = '../pretrained-word-embeddings/glove.6B/glove.6B.50d.txt'


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


if __name__ == '__main__':

    print(f'* try to load pretrained word vectors...')
    embedding, word2idx, idx2word, vocab_size, embedding_size = load_word_vectors(glove_filepath)
    print(f'* loading pretrained word vectors success!')

    def test(word, n=6):
        # embedding shape = (vocab_size, embedding_size)
        # but
        # embedding[i] shape = (embedding_size, )
        # should reshape it to (1, embedding_size)
        vector = embedding[word2idx[word]].reshape(1, embedding_size)
        # vector shape = (1, embedding_size)

        distances = pairwise_distances(vector, embedding, metric='cosine')
        # distances shape = (1, vocab_size)
        # should reshape it to (vocab_size, )
        distances = distances.reshape(vocab_size)

        idxs = distances.argsort()[1:n+1]
        print(f'=> words similar to "{word}":')
        for idx in idxs:
            print(f'{idx2word[idx]} - {distances[idx]}')
        print('')

    print(f'*** [Test] find most similar words ***')

    test('two')
    test('that')
    test('his')
    test('were')
    test('all')
    test('area')
    test('east')
    test('himself')
    test('white')
    test('man')
