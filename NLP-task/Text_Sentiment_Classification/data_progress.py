import torch
from torch.utils import data
from torch import nn
from gensim.models import Word2Vec


def load_data(path, lable=True, train=True):
    if train:
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            lines = [line.strip('\n').split(' ') for line in lines]

        x = []
        if lable:
            y = []
            for line in lines:
                y.append(line[0])
                x.append(line[2:])
            return x, y
        else:
            return lines
    else:
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            x = ["".join(line.strip('\n').split(",")[1:]).strip() for line in lines[1:]]
            x = [sen.split(' ') for sen in x]
        return x


# 构建数据集类
class TwitterDataset(data.Dataset):
    """
    Expected data shape like:(data_num, data_len)
    Data can be a list of numpy array or a list of lists
    input data shape : (data_num, seq_len, feature_dim)

    __len__ will return the number of data
    """

    def __init__(self, X, y):
        self.data = X
        self.label = y

    def __getitem__(self, idx):
        if self.label is None:
            return self.data[idx]
        return self.data[idx], self.label[idx]

    def __len__(self):
        return len(self.data)


# 数据预处理类
class Preprocess:
    def __init__(self, sen_len, w2v_path="./w2v.model"):
        self.embedding = Word2Vec.load(w2v_path)  # 把之前训练好的 wordtovec 模型读进来
        self.embedding_dim = self.embedding.vector_size
        self.sen_len = sen_len
        self.idx2word = []
        self.word2idx = {}
        self.embedding_matrix = []

    def add_embedding(self, word):
        # 把 word 加进 embedding，并赋予一个随机生成的 representation vector
        # word 只会是 "<PAD>" 或 "<UNK>"
        vector = torch.empty(1, self.embedding_dim)
        torch.nn.init.uniform_(vector)
        self.word2idx[word] = len(self.word2idx)
        self.idx2word.append(word)
        self.embedding_matrix = torch.cat([self.embedding_matrix, vector], 0)

    def make_embedding(self):
        # 制作一個 word2idx 的 dictionary
        # 制作一個 idx2word 的 list
        # 制作一個 word2vector 的 list
        for i, word in enumerate(self.embedding.wv.vocab):
            print('get words #{}'.format(i + 1), end='\r')
            self.word2idx[word] = len(self.word2idx)
            self.idx2word.append(word)
            self.embedding_matrix.append(self.embedding[word])
        print('')
        self.embedding_matrix = torch.tensor(self.embedding_matrix)
        # 将 "<PAD>" 跟 "<UNK>" 加进 embedding
        self.add_embedding("<PAD>")
        self.add_embedding("<UNK>")
        print("total words: {}".format(len(self.embedding_matrix)))
        return self.embedding_matrix

    def pad_sequence(self, sentence):
        # 将每个句子变成一样的长度
        if len(sentence) > self.sen_len:
            sentence = sentence[:self.sen_len]
        else:
            pad_len = self.sen_len - len(sentence)
            for _ in range(pad_len):
                sentence.append(self.word2idx["<PAD>"])
        assert len(sentence) == self.sen_len
        return sentence

    def sentence_word2idx(self, sentences):
        # 把每个句子的word转成相应的index
        sentence_list = []
        for i, sen in enumerate(sentences):
            print('sentence count #{}'.format(i + 1), end='\r')
            sentence_idx = []
            for word in sen:
                if (word in self.word2idx.keys()):
                    sentence_idx.append(self.word2idx[word])
                else:
                    sentence_idx.append(self.word2idx["<UNK>"])
            # 將每個句子變成一樣的長度
            sentence_idx = self.pad_sequence(sentence_idx)
            sentence_list.append(sentence_idx)
        return torch.LongTensor(sentence_list)

    def labels_to_tensor(self, y):
        # 把 labels 转成 tensor
        y = [int(label) for label in y]
        return torch.FloatTensor(y)
