import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F

import collections
import json
import logging
import os
from os.path import join
from nltk import tokenize

# In this package.
from bills.data import vocab
from bills import consts

def _status_to_class_idx(status):
    if status.startswith("ENACTED"):
        return 1
    else:
        return 0

class BillDataSet(Dataset):
    DOCUMENT = "document.txt"
    LABEL = "label.json"
    def __init__(self, root_dir, examples_file, transform=None):
        self.transform = transform
        with open(examples_file) as fp:
            self.bills = [os.path.join(root_dir, line.strip()) for line in fp]

    def __len__(self):
        return len(self.bills)

    def __getitem__(self, idx):
        billdir = self.bills[idx]
        with open(join(billdir, BillDataSet.DOCUMENT)) as fp:
            document = fp.read()

        with open(join(billdir, BillDataSet.LABEL)) as fp:
            label = json.load(fp)
            status = label['status']
            class_idx = _status_to_class_idx(label['status'])

        sample = {"document": document,
                  "status": status,
                  "label": class_idx }
        if self.transform:
            sample = self.transform(sample)

        return sample

def load_vocab(N=10000):
    filename = os.path.join(PROCESSED_DATA, "vocab.txt")
    with open(filename) as fp:
        tokens = [line.strip() for line in fp]

    return tokens[0:N]

class InputLayer(object):
    def __init__(self, vocab_file, vocab_size):
        logging.info("Loading vocabulary...")
        self.vocab = vocab.BaseVocabulary(vocab_file)
        assert self.vocab.PAD_ID == 0, "PAD ID is not zero!"
        logging.info("done.")
        self.trunc_vocab = self.vocab.ordered_tokens()[0:vocab_size]
        self.word2id = {value: idx for idx, value in enumerate(self.trunc_vocab)}

    def to_long_tensor(self, x):
        """ Converts batch of document text into LongTensor of shape (batch, M, N) where M is the max
        number of sentences in the batch and N is the max sentence length.
        """
        batch_size = len(x)
        docs = [[self.vocab.tokenize(self.vocab.normalize(sent))
                 for sent in tokenize.sent_tokenize(doc)] for doc in x]
        max_num_sentences = max(len(doc) for doc in docs)
        max_sent_length = max(len(sent) for doc in docs for sent in doc)
        ids = torch.LongTensor(batch_size, max_num_sentences, max_sent_length).zero_()
        for i, doc in enumerate(docs):
            for j, sent in enumerate(doc):
                for k, token in enumerate(sent):
                    ids[i, j, k] = self.word2id.get(token, self.vocab.OOV_ID)

        return ids

def _matrix_dims(size):
    N = 1
    for dim in size[0:-1]:
        N *= dim
    return (N, size[-1])

class AttentionReduction(torch.nn.Module):
    def __init__(self, input_dim, attention_dim):
        super(AttentionReduction, self).__init__()
        self.linear = torch.nn.Linear(input_dim, attention_dim)
        self.context = Variable(torch.randn(attention_dim, 1))

    def forward(self, matrix):
        """
        Args:
            matrix:  (N, M, input_dim)
        """
        x = F.tanh(self.linear(matrix))
        logging.debug("After tanh: %s", x.size())
        y = torch.mm(x.resize(*_matrix_dims(x.size())), self.context).resize(*x.size()[0:-1])
        logging.debug("After MM with context: %s", y.size())
        att = F.softmax(y)
        logging.debug("Att. shape: %s", att.size())
        att = torch.unsqueeze(att, -1)
        # element-wise, broadcast multiplication
        weighted_matrix = matrix * att
        reduced_vec = torch.sum(weighted_matrix, 1)
        logging.debug("Reduced vec shape: %s", reduced_vec.size())
        return reduced_vec


class HierarchicalAttention(torch.nn.Module):
    def __init__(self, embedding_dim, hidden_size, vocab_size, word_attention_dim=16,
                 sentence_attention_dim=32):
        super(HierarchicalAttention, self).__init__()
        # Embedding matrix for tokens.
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)

        # Parameters for creating sentence embeddings
        self.word_lstm = torch.nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=1,
                                       bidirectional=True, batch_first=True)
        # Note, the embedding is going to be 2*hidden_size because we are using a bi-directional
        # LSTM.
        self.word_attention = AttentionReduction(2*hidden_size, word_attention_dim)

        # And for creating full document embeddings.
        self.sent_lstm = torch.nn.LSTM(input_size=2*hidden_size, hidden_size=hidden_size, num_layers=1,
                                       bidirectional=True, batch_first=True)
        self.sentence_attention = AttentionReduction(2*hidden_size, sentence_attention_dim)

    def forward(self, x):
        logging.debug("Input shape: %s", x.size())
        flat_doc = x.resize(*_matrix_dims(x.size()))

        # How would we embed a single doc?
        x_embed = self.embedding(flat_doc)
        hidden, out = self.word_lstm(x_embed)
        logging.debug("Hidden shape: %s", hidden.size())
        # Now 'hidden' encodes each of the words in the context of their sentence.
        # Reduce with an attention mechanism.
        sentence_embed = self.word_attention(hidden)
        logging.debug ("Sentence embedding shape: %s", sentence_embed.size())

        # Un-flatten to process sentences now.
        sentence_embed = sentence_embed.view(x.size()[0], x.size()[1], -1)

        hidden, out = self.sent_lstm(sentence_embed)
        logging.debug("Sentence encoding shape: %s", hidden.size())
        doc_embed = self.sentence_attention(hidden)
        logging.debug("Doc embedding shape: %s", doc_embed.size())
        return doc_embed


def smoke_test():
    logging.basicConfig(level=logging.DEBUG)
    dset = BillDataSet(consts.PROCESSED_DATA_DIR, os.path.join(consts.PROCESSED_DATA_DIR, "train.txt"))
    dataloader = DataLoader(dset, batch_size=4, shuffle=True, num_workers=4)

    VOCAB_SIZE = 100
    WORD_EMBED_DIM = 50
    HIDDEN_SIZE = 8
    inputs = InputLayer(os.path.join(consts.PROCESSED_DATA_DIR, "vocab.txt"), VOCAB_SIZE)
    ha = HierarchicalAttention(WORD_EMBED_DIM, HIDDEN_SIZE, VOCAB_SIZE)

    NUM_CLASSES = 4
    linear = torch.nn.Linear(HIDDEN_SIZE*2, NUM_CLASSES)
    for batch in dataloader:
        x = inputs.to_long_tensor(batch['document'])
        logging.debug(x.size())
        out = ha(Variable(x))
        logging.debug("Output from HA: %s", out)
        logits = linear(out)
        probs = F.softmax(logits)
        print(batch['status'])
        loss = F.cross_entropy(logits, batch['status'])
        logging.info("Loss: %s", loss)


if __name__ == "__main__":
    smoke_test()
