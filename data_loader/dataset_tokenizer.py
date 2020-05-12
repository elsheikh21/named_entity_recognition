from collections import Counter
import re
import string
import nltk
from nltk.corpus import stopwords

alphanum = re.compile('^[a-zA-Z0-9_]+$')


class MyTokenizer(object):
    def __init__(self, keepStopwords=False, keepNum=False, keepAlphaNum=False, lower=True, minlength=0,
                 vocabSize=5000, minfreq=10e-5, stopset=None, vocab=None):

        self.keepStopwords = keepStopwords
        self.keepNum = keepNum
        self.keepAlphaNum = keepAlphaNum
        self.lower = lower
        self.minlength = minlength
        self.vocabSize = vocabSize
        self.minfreq = minfreq
        self.vocab = vocab
        self.keepStopwords = keepStopwords
        if not self.keepStopwords and not stopset:
            nltk.download('stopwords', quiet=True)
            stopset = set(stopwords.words('english') +
                          [p for p in string.punctuation])
        self.stopset = stopset

    def tokenize(self, text):
        if not self.lower:
            return text.split()
        else:
            return [t.lower() for t in text.split()]

    def get_vocab(self, Tokens):
        Vocab = Counter()
        for tokens in Tokens:
            Vocab.update(tokens)
        self.vocab = Counter(Vocab)

    def cleanTokens(self, Tokens):
        tokens_n = sum(self.vocab.values())
        filtered_voc = self.vocab.most_common(self.vocabSize)
        Freqs = Counter({t: f / tokens_n for t, f in filtered_voc if
                         f / tokens_n > self.minfreq and
                         t not in self.stopset
                         })
        words = list(Freqs.keys())
        # remove tokens that contain numbers
        if not self.keepAlphaNum and not self.keepNum:
            alpha = re.compile('^[a-zA-Z_]+$')
            words = [w for w in words if alpha.match(w)]
        # or just remove tokens that contain a combination of letters and numbers
        elif not self.keepAlphaNum:
            alpha_or_num = re.compile('^[a-zA-Z_]+|[0-9_]+$')
            words = [w for w in words if alpha_or_num.match(w)]
        words.sort()
        self.words = words
        words2idx = {w: i for i, w in enumerate(words)}
        self.words2idx = words2idx
        print('Vocabulary')
        print(words[:12])
        cleanTokens = []
        for tokens in Tokens:
            cleanTokens.append([t for t in tokens if t in words])
        self.tokens = cleanTokens


if __name__ == '__main__':
    tokenizer = MyTokenizer()
    Tokens = [tokenizer.tokenize(text) for text in MyDatasetReader(dataset_path)]
    tokenizer.get_vocab(Tokens)
    print('Most frequent words')
    print(tokenizer.vocab.most_common(12))
    tokenizer.cleanTokens(Tokens)
