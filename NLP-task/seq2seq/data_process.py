import json
import os
import re

import torch
import torch.utils.data as data
import numpy as np


class Dataset(data.Dataset):
    def __init__(self, root_path, max_len, set_name):
        self.root_path = root_path
        self.set_name = set_name
        self.max_len = max_len

        self.word2int_cn, self.int2word_cn = self.get_dictionary('cn')
        self.word2int_en, self.int2word_en = self.get_dictionary('en')

        self.data = self.get_data()

        self.cn_vocab_size = len(self.word2int_cn)
        self.en_vocab_size = len(self.word2int_en)

        # self.padding = padding(max_len, self.word2int_en['<PAD>'])

    # 读入中文、英文词典
    def get_dictionary(self, language_name):
        with open(os.path.join(self.root_path, f'word2int_{language_name}.json'), 'r', encoding='utf-8') as f:
            word2int = json.load(f)
        with open(os.path.join(self.root_path, f'int2word_{language_name}.json'), 'r', encoding='utf-8') as f:
            int2word = json.load(f)
        return word2int, int2word

    # 读入数据集
    def get_data(self):
        data = []
        with open(os.path.join(self.root_path, f'{self.set_name}.txt'), 'r', encoding='utf-8') as f:
            for line in f:
                data.append(line)
        print(f'{self.set_name} dataset size: ', {len(data)})
        return data

    def padding(self, x):
        x = np.pad(x, (0, (self.max_len - x.shape[0])), mode='constant', constant_values=self.word2int_en['<PAD>'])
        return x

    def __len__(self):
        return len(self.data)

    def __getitem__(self, Index):
        # 先將中英文分开
        sentences = self.data[Index]
        sentences = re.split('[\t\n]', sentences)
        sentences = list(filter(None, sentences))
        # print (sentences)
        assert len(sentences) == 2

        # 特殊字符
        BOS = self.word2int_en['<BOS>']
        EOS = self.word2int_en['<EOS>']
        UNK = self.word2int_en['<UNK>']

        # 在开头添加 <BOS>，在结尾添加 <EOS> ，不在字典的 subword (词) 用 <UNK> 取代
        en, cn = [BOS], [BOS]
        # 將英文句子拆解为 subword 并转成整数
        # e.g. < BOS >, we, are, friends, < EOS > --> 1, 28, 29, 205, 2
        sentence = re.split(' ', sentences[0])
        sentence = list(filter(None, sentence))
        for word in sentence:
            en.append(self.word2int_en.get(word, UNK))
        en.append(EOS)

        # 将中文句子分词，并用字典对应的index代替分词
        sentence = re.split(' ', sentences[1])
        sentence = list(filter(None, sentence))
        for word in sentence:
            cn.append(self.word2int_cn.get(word, UNK))
        cn.append(EOS)

        en, cn = np.asarray(en), np.asarray(cn)

        # 用 <PAD> 將句子补到相同長度
        en, cn = self.padding(en), self.padding(cn)
        en, cn = torch.LongTensor(en), torch.LongTensor(cn)

        return en, cn
