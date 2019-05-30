import numpy as np
from brown_corpus import get_sentences_with_word2idx_limit_vocab, get_sentences_with_word2idx

def get_bigram_probs(sentences, n_vocab, start_token_idx, end_token_idx, smoothing=1):

    # add-one smoothing
    # consider p(dog|the quick brown fox jumps over the)
    # "The quick brown fox jumps over the lazy dog" is probably the only sentence like this in our corpus
    # we know that "The quick brown fox jumps over the lazy turtle" is a valid sentence too
    # but we may find by counting that p(turtle|the quick brown fox jumps over the) = 0
    # 0 isn't accurate because we know this sentence makes sense, our language model should allow for it
    #
    # add a small number to each count
    # and divide by vocabulary size to ensure probabilities sum to 1
    bigram_probs = np.ones((n_vocab, n_vocab)) * smoothing

    for sent in sentences:
        for i in range(len(sent)):

            # add start_token -> the 1st word prob
            if i == 0:
                bigram_probs[start_token_idx, sent[i]] += 1
            # add the last word -> end_token prob
            elif i == len(sent) - 1:
                bigram_probs[sent[i], end_token_idx] += 1
            else:
                bigram_probs[sent[i-1], sent[i]] += 1

    bigram_probs /= bigram_probs.sum(axis=1, keepdims=True)
    return bigram_probs


if __name__ == '__main__':

    # the sentences have already been converted to word indexes
    # and the out-of-vocabulary words have been converted to 'UNKNOWN' token
    indexed_sents, word2idx = get_sentences_with_word2idx_limit_vocab(10000)

    n_vocab = len(word2idx)
    print(f'Vocab size: {n_vocab}')

    start_token_idx = word2idx['START']
    end_token_idx = word2idx['END']

    bigram_probs = get_bigram_probs(indexed_sents,
                                    n_vocab,
                                    start_token_idx,
                                    end_token_idx,
                                    smoothing=1)


    def get_log_score(sent):
        score = 0
        for i in range(len(sent)):
            # get prob of start token -> the 1st word
            if i == 0:
                score += np.log(bigram_probs[start_token_idx, sent[i]])
            else:
                score += np.log(bigram_probs[sent[i - 1], sent[i]])
        # get prob of the last word -> end token
        score += np.log(bigram_probs[sent[-1], end_token_idx])

        # normalize the log score
        return score / (len(sent) + 1)

    # to get original sentence from indexed sentence
    idx2word = dict((v, k) for k, v in word2idx.items())

    def get_words_from_idx(sent):
        return ' '.join(idx2word[i] for i in sent)

    # generate an uniform distribution fake bigram_probs
    # to sample a fake sentence
    # to compare the log scores of real sentence and fake sentence
    fake_bigram_probs = np.ones(n_vocab)
    fake_bigram_probs[start_token_idx] = 0
    fake_bigram_probs[end_token_idx] = 0
    fake_bigram_probs /= fake_bigram_probs.sum()

    while True:
        # get a random real sentence
        rand = np.random.choice(len(indexed_sents))
        real_sent = indexed_sents[rand]

        # get a random fake sentence
        fake_sent = np.random.choice(n_vocab, size=len(real_sent), p=fake_bigram_probs)

        # the score of real sentence would always higher than the score of fake sentence
        print(f'Real sentence: {get_words_from_idx(real_sent)} => Score: {get_log_score(real_sent)}')
        print(f'Fake sentence: {get_words_from_idx(fake_sent)} => Score: {get_log_score(fake_sent)}')

        cont = input("Continue? [Y/n]")
        if cont and cont.lower() in ('N', 'n'):
            break
