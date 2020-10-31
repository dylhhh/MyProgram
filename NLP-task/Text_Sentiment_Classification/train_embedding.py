# 使用数据集训练wor2vec词向量
import logging
import multiprocessing
import os
import sys

from gensim.models import Word2Vec
from gensim.models.word2vec import PathLineSentences

from data_progress import load_data
import jieba
import jieba.analyse


# model = Word2Vec(sentences, sg=1, size=100,  window=5,  min_count=5,  negative=3, sample=0.001, hs=1, workers=4)

# 分词
def segment(file1, file2, file3):
    segment_file = open('./data/segment.txt', 'a', encoding='utf8')
    with open(file1, encoding='utf8') as f:
        text = f.readlines()
        for sentence in text:
            sentence = list(jieba.cut(sentence))
            segment_file.write(" ".join(sentence))
        del text
        f.close()
    with open(file2, encoding='utf8') as f:
        text = f.readlines()
        for sentence in text:
            sentence = list(jieba.cut(sentence))
            segment_file.write(" ".join(sentence))
        del text
        f.close()
    with open(file3, encoding='utf8') as f:
        text = f.readlines()
        for sentence in text:
            sentence = list(jieba.cut(sentence))
            segment_file.write(" ".join(sentence))
        del text
        f.close()
    segment_file.close()

if __name__ == '__main__':
    # 分词
    # segment('./data/training_nolabel.txt', './data/training_label.txt', './data/testing_data.txt')

    # 日志信息输出
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))

    input_file = './data/segment.txt'
    outp1 = './data/embedding1.model'
    outp2 = './data/word2vec_format'
    # fileNames = os.listdir(input_dir)
    # 训练模型
    # 输入语料目录:PathLineSentences(input_dir)
    # embedding size:256 共现窗口大小:10 去除出现次数5以下的词,多线程运行,迭代10次
    model = Word2Vec(PathLineSentences(input_file),
                     size=256, window=10, min_count=5,
                     workers=multiprocessing.cpu_count(), iter=5)
    model.save(outp1)
    model.wv.save_word2vec_format(outp2, binary=False)



