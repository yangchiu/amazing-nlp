import os
from pathlib import Path
from datetime import datetime

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
from opinrank_corpus import get_sentences

savedir = 'word2vec_gensim'
model_filename = 'word2vec.model'
vector_filename = 'word2vec.model.txt'

# FAST_VERSION reflects the return type of your BLAS interface (single vs. double).
# Both 0 and 1 should be fine, it's an internal detail.
print(f'gensim.models.word2vec.FAST_VERSION = {FAST_VERSION}')


def train_model():
    sentences = get_sentences()

    t0 = datetime.now()
    print(f'=> start training...')
    # size = word vector dimension
    # window = context distance. the max distance between the target word and
    #          its neighboring word.
    # min_count = if the frequent count of a word is lower than min_count,
    #             the model will ignore it.
    # workers = number of threads to be used
    model = Word2Vec(sentences, size=150, window=10, min_count=2, workers=10)
    model.train(sentences, total_examples=len(sentences), epochs=10)
    print(f'=> training time {datetime.now() - t0}')

    # save the model
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    model.save(f'{savedir}/{model_filename}')
    model.wv.save_word2vec_format(f'{savedir}/{model_filename}', binary=False)

    return model


def load_model():
    model = Word2Vec.load(f'{savedir}/{model_filename}')
    return model


def test_model(model):

    print(f'*** [Test] find most similar words ***')
    w1 = 'dirty'
    print(f'=> words similar to "{w1}":')
    list = model.wv.most_similar(positive=w1, topn=5)
    for elem in list:
        print(f'{elem[0]} - {elem[1]}')
    print('')

    w1 = 'polite'
    print(f'=> words similar to "{w1}":')
    list = model.wv.most_similar(positive=w1, topn=5)
    for elem in list:
        print(f'{elem[0]} - {elem[1]}')
    print('')

    w1 = 'france'
    print(f'=> words similar to "{w1}":')
    list = model.wv.most_similar(positive=w1, topn=5)
    for elem in list:
        print(f'{elem[0]} - {elem[1]}')
    print('')


if __name__ == '__main__':
    if Path(f'{savedir}/{model_filename}').is_file():
        print(f'=> load old model...')
        model = load_model()
    else:
        print(f'=> train new model...')
        model = train_model()
    test_model(model)
