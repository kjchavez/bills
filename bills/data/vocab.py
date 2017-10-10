""" Utilities for creating vocabulary.
"""
import codecs
import collections
import glob
import os
import logging
import unicodedata
import sys

from bills import consts

def _freq(key, counter):
    return float(counter[key]) / sum(counter.values())

def _file_contents_generator(filepattern):
    for filename in glob.glob(filepattern):
        with codecs.open(filename, encoding='utf-8') as fp:
            yield filename, fp.read()

# TODO(kjchavez): Separate the freq based filtering from the Vocab class.
class Vocabulary(object):
    OOV = "<oov>"

    # Newline characters don't saveto() file nicely, so we use this placeholder
    # for the character.
    NEWLINE = "<newline>"

    def __init__(self, tokens, counts):
        self.word2id = {}
        self.word2count = counts
        self.id2word = [Vocabulary.OOV]
        self.word2id[Vocabulary.OOV] = 0
        idx = 1
        for token in tokens:
            if token == Vocabulary.NEWLINE:
                token = u'\n'

            self.word2id[token] = idx
            self.id2word.append(token)
            idx += 1

    def ordered_tokens(self):
        return self.id2word

    def saveto(self, filename):
        with codecs.open(filename, 'w', encoding='utf-8') as fp:
            for word in self.ordered_tokens():
                if word == "\n":
                    word = Vocabulary.NEWLINE

                try:
                    fp.write(word)
                except:
                    logging.warning("Error in saving vocabulary.")
                    logging.warning("Offending token: %s", word)
                    logging.warning("Individual chars: %s", [c for c in word])
                    logging.warning("If tokens are coming from file, check encoding!")
                    sys.exit(1)

                fp.write('\n')

    def get(self, token):
        """ Returns id of token, or id for <oov> if token is unknown. """
        return self.word2id.get(token,
                self.word2id[Vocabulary.OOV])

    def get_token(self, idx):
        return self.id2word[idx]

    def size(self):
        return len(self.word2id)

    @staticmethod
    def fromfile(filename):
        with codecs.open(filename, encoding='utf-8') as fp:
            oov = next(fp).strip()
            assert oov == Vocabulary.OOV
            return Vocabulary(tokens=[line.strip() for line in fp],
                              counts=collections.Counter())

    @staticmethod
    def fromiterator(iterator, tokenize_fn,
                     max_num_tokens=None, min_freq=None, min_count=None,
                     extra=[]):
        """ Builds vocab from an iterator that yields (id, text) tuples."""
        counter = collections.Counter()
        for idx, text in iterator:
            print "Processing:", idx
            counter.update(tokenize_fn(text))

        tokens = extra
        for key, count in counter.most_common(max_num_tokens):
            if (min_freq and _freq(key, counter) < min_freq) or \
               (min_count and count < min_count):
                break
            tokens.append(key)

        return Vocabulary(tokens=tokens, tokenize_fn=tokenize_fn)

    @staticmethod
    def build(filepattern, tokenize_fn, max_num_tokens=None, min_freq=None, min_count=None):
        return Vocabulary.fromiterator(_file_contents_generator(filepattern),
                                       tokenize_fn=tokenize_fn,
                                       max_num_tokens=max_num_tokens,
                                       min_freq=min_freq,
                                       min_count=min_count)

if __name__ == "__main__":
    def tokenize(x):
        return x.lower().split()

    v = Vocabulary.build(os.path.join(consts.PROCESSED_DATA_DIR, "*", "document.txt"), tokenize)
    v.saveto(os.path.join(consts.PROCESSED_DATA_DIR, "vocab.txt"))
