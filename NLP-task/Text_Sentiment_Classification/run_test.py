import os

import torch
import pandas as pd
from torch.utils.data import DataLoader
from data_progress import load_data, TwitterDataset, Preprocess
from training_testing import testing


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 参数设置
    sen_len = 20
    batch_size = 128
    wv_path = "./data/embedding1.model"
    model_path = './model'

    # 加载数据
    test_data = load_data('./data/testing_data.txt', lable=False, train=False)
    # 数据预处理
    preprocess = Preprocess(sen_len, wv_path)
    embedding = preprocess.make_embedding()
    test_x = preprocess.sentence_word2idx(test_data)
    # 数据集
    test_dataset = TwitterDataset(X=test_x, y=None)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

    # 加载模型
    model = torch.load(os.path.join(model_path, 'ckpt_6449.443517981439.model'))
    # 测试
    test_pred, _ = testing(model, device, test_loader)
    test_pred = test_pred.view(-1).tolist()

    # 写入csv文件
    tmp = pd.DataFrame({"id": [str(i) for i in range(len(test_x))], "label": list(map(int, test_pred))})
    print("save csv ...")
    tmp.to_csv(r'./predict.csv', index=False)
    print("Finish Predicting")
