import os
from opinrank_corpus import get_sentences
from gensim.models import Word2Vec
from gensim.models.word2vec import FAST_VERSION
# if you encounter:
#
# UserWarning: C extension not loaded, training
# will be slow. Install a C compiler and reinstall gensim for fast training.
# "C extension not loaded, training will be slow. "
#
# (1) install mingw and add it to path
# or
# (2) downgrade gensim version from 3.7.3 to 3.6
#
# FAST_VERSION reflects the return type of your BLAS interface (single vs. double).
# Both 0 and 1 should be fine, it's an internal detail.
print(f'gensim.models.word2vec.FAST_VERSION = {FAST_VERSION}')

save_dir = 'word2vec_gensim/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

model_filename = 'word2vec.model'
vector_filename = 'word2vec.model.txt'


def train_model():

    sentences = get_sentences()

    print(f'=> start training...')
    # size = word vector dimension
    # window = context distance. the max distance between the target word and
    #          its neighboring word.
    # min_count = if the frequent count of a word is lower than min_count,
    #             the model will ignore it.
    # workers = number of threads to be used
    model = Word2Vec(sentences, size=128, window=10, min_count=2, workers=10)
    model.train(sentences, total_examples=len(sentences), epochs=10)

    # save the model
    model.save(os.path.join(save_dir, model_filename))
    model.wv.save_word2vec_format(os.path.join(save_dir, vector_filename), binary=False)

    return model


def load_model():
    model = Word2Vec.load(os.path.join(save_dir, model_filename))
    return model


def test_model(model):

    print(f'*** [Test] find most similar words ***')

    def test(word):
        results = model.wv.most_similar(positive=word, topn=6)
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

    if os.path.exists(os.path.join(save_dir, model_filename)):
        print(f'=> load existing model...')
        model = load_model()
    else:
        print(f'=> train new model...')
        model = train_model()

    test_model(model)
