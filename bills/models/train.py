import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchtext

import collections
import json
import os
from os.path import join

# In this package.
from bills.data import vocab
from bills import consts

project_dir = join(os.path.dirname(__file__), os.pardir, os.pardir)
PROCESSED_DATA = join(project_dir, "data", "processed")

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
            status = label["status"]

        sample = {"document": document,
                  "status": status }
        if self.transform:
            sample = self.transform(sample)

        return sample

def load_vocab(N=10000):
    filename = os.path.join(PROCESSED_DATA, "vocab.txt")
    with open(filename) as fp:
        tokens = [line.strip() for line in fp]

    return tokens[0:N]

dset = BillDataSet(PROCESSED_DATA, os.path.join(PROCESSED_DATA, "train.txt"))
dataloader = DataLoader(dset, batch_size=16, shuffle=True, num_workers=4)
for i_batch, batch in enumerate(dataloader):
    print(i_batch)
    print(batch['status'])
    break


class InputLayer(object):
    def __init__(self, vocab_file, vocab_size, embedding_dim):
        # self.vocab = load_vocab(N=vocab_size)
        self.vocab = vocab.Vocabulary.fromfile(vocab_file)
        self.trunc_vocab = self.vocab.ordered_tokens()[0:vocab_size]
        self.word2id = {value: idx for idx, value in enumerate(self.trunc_vocab)}
        self.embedding = torch.nn.Embedding(len(self.trunc_vocab), embedding_dim)

    def embed_id(self, long_tensor):
        return self.embedding(Variable(long_tensor))

    def embed_tokens(self, tokens):
        ids = [[self.word2id.get(t, 0) for t in instance] for instance in tokens]
        return self.embed_id(torch.LongTensor(ids))


def smoke_test():
    inputs = InputLayer(os.path.join(consts.PROCESSED_DATA_DIR, "vocab.txt"), 100, 50)
    print(inputs.embedding.weight)
    ids = torch.LongTensor([[0, 1], [ 6, 9]])
    x = inputs.embed_id(ids)
    print(x)

    words = ["for this section".split(), "shall be under".split()]
    x = inputs.embed_tokens(words)
    print(x)


#vocab = torchtext.vocab.Vocab(collections.Counter())
#vocab.load_vectors(torchtext.vocab.pretrained_aliases["glove.6B.50d"])
smoke_test()
