import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch import optim
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

# TODO(kjchavez): Add some safety constraints here.
#  * Add absolute maximum bound on document length.
class BillDataSet(Dataset):
    DOCUMENT = "document.txt"
    LABEL = "label.json"
    def __init__(self, root_dir, examples_file, transform=None, max_doc_len=10000):
        self.transform = transform
        self.max_doc_len = max_doc_len
        with open(examples_file) as fp:
            self.bills = [os.path.join(root_dir, line.strip()) for line in fp]

    def __len__(self):
        return len(self.bills)

    def __getitem__(self, idx):
        billdir = self.bills[idx]
        with open(join(billdir, BillDataSet.DOCUMENT)) as fp:
            document = fp.read()

        # Truncate if necessary. Makes sure to only keep whole words.
        if len(document) > self.max_doc_len:
            document = document[0:self.max_doc_len].rsplit(' ', 1)[0]

        with open(join(billdir, BillDataSet.LABEL)) as fp:
            label = json.load(fp)
            status = label['status']
            class_idx = torch.LongTensor([_status_to_class_idx(label['status'])])

        sample = {"document": document,
                  "status": status,
                  "label": class_idx }
        if self.transform:
            sample = self.transform(sample)

        return sample


class InputLayer(object):
    def __init__(self, vocab_file, vocab_size):
        logging.info("Loading vocabulary...")
        self.vocab = vocab.BaseVocabulary(vocab_file)
        assert self.vocab.PAD_ID == 0, "PAD ID is not zero!"
        logging.info("done.")
        self.trunc_vocab = self.vocab.most_common(vocab_size)
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

# Since torch matrix multiplication operations only work with 2D tensors, this little helper makes
# it convenient to work with higher dim tensors.
def _tensor_dot(t1, t2):
    """ Multiples an (N, *, K) tensor by a (K, M) matrix. Returns (N, *, M) tensor. """
    original_size = t1.size()
    x = t1.resize(*_matrix_dims(t1.size()))
    return torch.mm(x, t2).resize(*original_size[0:-1])

class AttentionReduction(torch.nn.Module):
    """ Reduces a matrix of embeddings to a single vector using attention.

    Takes an M x N x D matrix and reduces along the 2nd dimension using a weighted sum based on
    attention computed using a learned context vector.

    Returns:
        M x D matrix.
    """
    def __init__(self, input_dim, attention_dim):
        super(AttentionReduction, self).__init__()
        self.linear = torch.nn.Linear(input_dim, attention_dim)
        self.context = torch.nn.Parameter(torch.randn(attention_dim, 1))

    def forward(self, matrix):
        logging.debug("Input matrix size: %s", matrix.size())
        hidden = F.tanh(self.linear(matrix))
        y = _tensor_dot(hidden, self.context)
        logging.debug("After MM with context: %s", hidden.size())
        att = F.softmax(y)
        logging.debug("Attention weights size: %s", att.size())

        # To make the broadcast operation work, we need the number of dimensions to match.
        # matrix is M x N x D. att is M x N -- so we make it M x N x 1.
        att = torch.unsqueeze(att, -1)
        weighted_matrix = matrix * att

        summary = torch.sum(weighted_matrix, 1)
        logging.debug("Summary vector size: %s", summary.size())
        return summary


class HierarchicalAttention(torch.nn.Module):
    def __init__(self, embedding_dim, hidden_size, vocab_size, word_attention_dim=16,
                 sentence_attention_dim=32):
        super(HierarchicalAttention, self).__init__()
        # Embedding matrix for tokens.
        # NOTE(kjchavez): We probably want to keep this on the CPU, even if we move the rest of the
        # model to a GPU.
        # TODO(kjchavez): Clean way to do this?
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
        x_embed = self.embedding(flat_doc)

        # TODO(kjchavez): If we move the embedding matrix to the CPU, then we'll have to call,
        #
        #   x_embed = x_embed.cuda()

        # NOTE: We shouldn't run the lstm over the whole sequence, since lengths vary.
        # Use packed/padded utils.
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

class DocumentClassifier(torch.nn.Module):
    def __init__(self, word_embed_dim=50, hidden_size=8, vocab_size=100, num_classes=2):
        super(DocumentClassifier, self).__init__()
        self.hier_att = HierarchicalAttention(word_embed_dim, hidden_size, vocab_size)
        self.linear = torch.nn.Linear(hidden_size*2, num_classes)

    def forward(self, x):
        logits = self.linear(self.hier_att(x))
        logging.debug("Logits shape: %s", logits.size())
        return logits


def smoke_test():
    logging.basicConfig(level=logging.INFO)

    # Creates a dataset from the processed data that was created by bills/data/make_dataset.py.
    # Trying out the project structure from Cookie Cutter Data Science.
    dset = BillDataSet(consts.PROCESSED_DATA_DIR, os.path.join(consts.PROCESSED_DATA_DIR, "train.txt"))
    dataloader = DataLoader(dset, batch_size=4, shuffle=True, num_workers=4)

    # Hyper parameters for a sanity check.
    # TODO(kjchavez): What's the "right" way to work with hyperparameters in PyTorch?
    VOCAB_SIZE = 5000
    WORD_EMBED_DIM = 100
    HIDDEN_SIZE = 50

    inputs = InputLayer(os.path.join(consts.PROCESSED_DATA_DIR, "vocab.txt"), VOCAB_SIZE)
    model = DocumentClassifier(word_embed_dim=WORD_EMBED_DIM,
                              hidden_size=HIDDEN_SIZE,
                              vocab_size=VOCAB_SIZE)

    # In PyTorch, to move an entire nn.Module to the GPU, we call .cuda(). This moves all the
    # torch.nn.Parameters defined in the module (and submodules) to a GPU.
    # NOTE: From docs, this should be done prior to creating the optimizer for this model.
    model = model.cuda()

    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    for batch in dataloader:
        # We are intentionally not moving this to the GPU. The embedding matrix will live on the
        # CPU. Even though we move the 'model' to gpu.
        x = Variable(inputs.to_long_tensor(batch['document'])).cuda()
        logging.debug("Dimensions of input: %s", x.size())
        logits = model(x)
        probs = F.softmax(logits)
        label = Variable(batch['label'].squeeze(), requires_grad=False)
        loss = F.cross_entropy(logits, label.cuda())
        logging.info("Loss: %s", loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


if __name__ == "__main__":
    smoke_test()
