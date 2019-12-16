import numpy as np
import torch
from fasttext import load_model
from functools import wraps
from torch.autograd import Variable
from torch.nn.modules.sparse import EmbeddingBag
from torch.utils.data import Dataset, DataLoader


def singleton(cls):
    instance = None

    @wraps(cls)
    def inner(*args, **kwargs):
        nonlocal instance
        if instance is None:
            instance = cls(*args, **kwargs)
        return instance

    return inner


class LanguageModelData(Dataset):
    def __init__(self, data, max_len, pad_index, eos_index):
        self.data = data
        self.max_len = max_len
        self.pad_index = pad_index
        self.eos_index = eos_index

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sequence = self.data[index][: self.max_len]
        x = sequence[:]
        y = sequence[1:] + [self.eos_index]
        assert len(x) == len(y)
        pad = [self.pad_index] * (self.max_len - len(x))
        x = torch.tensor(x + pad).long()
        y = torch.tensor(y + pad).long()
        return x, y


@singleton
class EmbeddingBuilder(object):
    """
  builds word2index dictionary and embeddings array from the model_path of .vec file format
  """

    def __init__(self, model_path):
        self.model_path = model_path
        self.word2index = {
            token: i for i, token in enumerate(("<PAD>", "<SOS>", "<EOS>"))
        }
        self.embeddings = []
        self.vocab_size = None
        self.embedding_dim = None
        self._build()

    def _build(self):
        embedding_model = open(self.model_path)
        vocab_size, embedding_dim = embedding_model.readline().split()
        self.vocab_size, self.embedding_dim = int(vocab_size), int(embedding_dim)
        self.embeddings.append(np.zeros((1, self.embedding_dim)))
        while True:
            line = fasttext_model.readline().strip()
            if not line:
                break
            curr_splits = line.split()
            curr_word = " ".join(curr_splits[: -self.embedding_dim])
            if curr_word not in self.word2index:
                self.word2index[curr_word] = len(self.word2index)
                curr_embedding = curr_splits[-self.embedding_dim :]
                curr_embedding = np.expand_dims(
                    np.array(list(map(float, curr_embedding))), 0
                )
                self.embeddings.append(curr_embedding)
        embedding_model.close()
        self.embeddings = np.concatenate(self.embeddings)


@singleton
class FastTextVectorizer(object):
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = load_model(self.model_path)
        self.input_matrix = self.model.get_input_matrix()
        self.matrix_shape = self.input_matrix.shape
        self.embedding_bag = EmbeddingBag(
            self.matrix_shape[0], self.matrix_shape[1], mode="mean"
        )
        self.embedding_bag.weight.data.copy_(torch.FloatTensor(self.input_matrix))

    def forward(self, tokens):
        token_subindexes = np.empty([0], dtype=np.int64)
        token_offsets = [0]
        for token in tokens:
            _, subinds = self.model.get_subwords(token)
            token_subindexes = np.concatenate((token_subindexes, subinds))
            token_offsets.append(token_offsets[-1] + len(subinds))
        token_offsets = token_offsets[:-1]
        ind = Variable(torch.LongTensor(token_subindexes))
        offsets = Variable(torch.LongTensor(token_offsets))
        return self.embedding_bag.forward(ind, offsets)
