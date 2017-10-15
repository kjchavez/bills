""" Utilities for creating vocabulary.
"""
# import spacy
import argparse
import codecs
import collections
import glob
import os
import logging
import unicodedata
import sys

from bills import consts

class BaseVocabulary(object):
    """ Base class for vocabularies.

    All subclasses should provide a normalize() and a tokenize() function.
    """
    PAD = "<pad>"
    PAD_ID = 0
    OOV = "<oov>"
    OOV_ID = 1
    def __init__(self, filename):
        self.filename = filename
        self.word2id = {BaseVocabulary.PAD: 0, BaseVocabulary.OOV: 1}
        self.id2word = {0: BaseVocabulary.PAD, 1: BaseVocabulary.OOV}
        self.word2count = collections.Counter({token: 0 for token in self.word2id.keys()})
        self.next_id = max(self.id2word.keys()) + 1
        self.load(filename)

    def normalize(self, text):
        return text.replace('\n', "<newline>").replace('\t', "<tab>")

    def tokenize(self, text):
        return text.split()

    def consume_tokens(self, tokens):
        for token in tokens:
            if token not in self.word2id:
                self.word2id[token] = self.next_id
                self.id2word[self.next_id] = token
                self.next_id += 1

        self.word2count.update(tokens)

    def consume_texts(self, iterator):
        """ Consumes tokens from texts returned from |iterator|. """
        for idx, text in enumerate(iterator):
            if (idx+1) % 100 == 0:
                logging.info("Processed %d texts.", idx+1)
            tokens = self.tokenize(self.normalize(text))
            self.consume_tokens(tokens)

        self.save(self.filename)

    def ordered_tokens(self):
        return [self.id2word[i] for i in range(len(self.id2word))]

    def most_common(self, n):
        return self.word2count.most_common(n)

    def load(self, filename):
        if not os.path.exists(filename):
            logging.info("Vocab file does not exist. Will create new one.")
            return

        with codecs.open(filename, encoding='utf-8') as fp:
            for line in fp:
                token, idx_str, count_str = line.split("<<>>")
                idx = int(idx_str)
                count = int(count_str)
                self.word2id[token] = idx
                self.id2word[idx] = token
                self.word2count[token] = count
                self.next_id = max(self.next_id, idx+1)

    def save(self, filename):
        """ Saves tokens from vocabulary to |filename|.

        File format:

            {token}<<>>{id}<<>>{count}
        """
        with codecs.open(filename, 'w', encoding='utf-8') as fp:
            for token in self.word2count.keys():
                line = "<<>>".join([token, str(self.word2id[token]), str(self.word2count[token])])
                try:
                    fp.write(line)
                    fp.write("\n")
                except:
                    logging.warning("Error in saving vocabulary.")
                    logging.warning("Offending token: %s", token)
                    logging.warning("Individual chars: %s", [c for c in token])
                    logging.warning("If tokens are coming from file, check encoding!")
                    sys.exit(1)

class SpacyVocab(BaseVocabulary):
    def __init__(self, filename):
        super(SpacyVocab, self).__init__(filename)
        self.nlp = spacy.load('en')

    def normalize(self, text):
        return super(SpacyVocab, self).normalize(text).lower()

    def tokenize(self, text):
        logging.debug("Tokenizing text of length: %s", len(text))
        if len(text) > 100000:
            logging.info("Large text. %d chars. Trying...", len(text))

        doc = self.nlp(text)
        if len(text) > 100000:
            logging.info("...success.")
        return [tok for tok in doc]


def _file_contents_generator(filepattern):
    for filename in glob.glob(filepattern):
        with codecs.open(filename, encoding='utf-8') as fp:
            yield fp.read()


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    create = subparsers.add_parser("create", help="build new vocabulary")
    create.add_argument("--vocabfile", default=os.path.join(consts.PROCESSED_DATA_DIR,
                                                            "vocab.txt"))
    create.add_argument("--reset", action="store_true", default=False)
    create.add_argument("--source", default=os.path.join(consts.PROCESSED_DATA_DIR, "*",
                                                          "document.txt"),
                        help="filepattern for source of text for vocab.")
    create.set_defaults(func=create_vocab)

    inspect = subparsers.add_parser("inspect", help="inspect existing vocabulary")
    inspect.add_argument("--vocabfile", default=os.path.join(consts.PROCESSED_DATA_DIR,
                                                            "vocab.txt"))
    inspect.set_defaults(func=inspect_vocab)
    return parser.parse_args()

def create_vocab(args):
    if args.reset and os.path.exists(args.vocabfile):
        os.remove(args.vocabfile)

    v = BaseVocabulary(args.vocabfile)
    v.consume_texts(_file_contents_generator(args.source))

def inspect_vocab(args):
    v = BaseVocabulary(args.vocabfile)
    print("Size:", len(v.ordered_tokens()))
    print("Top 25 tokens:")
    for token, count in v.word2count.most_common(25):
        print(token, count)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    args.func(args)
