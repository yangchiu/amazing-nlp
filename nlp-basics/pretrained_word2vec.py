from gensim.models import KeyedVectors

# https://code.google.com/archive/p/word2vec/
# direct download link: https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing
# 3 million words and phrases
# embedding size = 300
word2vec_filepath = '../pretrained-word-embeddings/w2v/GoogleNews-vectors-negative300.bin'


def test_word_vectors(word_vectors):

    print(f'*** [Test] find most similar words ***')

    def test(word):
        results = word_vectors.most_similar(positive=word, topn=6)
        print(f'=> words similar to "{word}":')
        for elem in results:
            print(f'{elem[0]} - {elem[1]}')
        print('')

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


if __name__ == '__main__':

    print(f'* try to load pretrained word vectors...')
    word_vectors = KeyedVectors.load_word2vec_format(word2vec_filepath, binary=True)
    print(f'* loading pretrained word vectors success!')
    test_word_vectors(word_vectors)
