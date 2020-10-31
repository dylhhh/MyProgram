import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from data_progress import load_data, TwitterDataset, Preprocess
from model import GRU
from training_testing import training, testing

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 参数设置
    sen_len = 20
    batch_size = 128
    total_epochs = 20
    train_epochs = 5
    lr = 0.001
    wv_path = "./data/embedding1.model"
    hidden_dim = 150
    num_layers = 2
    dropout = 0.5
    model_path = './model'
    best_acc = 0

    # 加载数据
    print("loading data...")
    train_x, y = load_data('./data/training_label.txt')
    train_nolabel = load_data('./data/training_nolabel.txt', lable=False)

    preprocess = Preprocess(sen_len, wv_path)
    embedding = preprocess.make_embedding()
    train_x_label = preprocess.sentence_word2idx(train_x)  # 有标签数据预处理
    train_y_label = preprocess.labels_to_tensor(y)
    train_x_nolabel = preprocess.sentence_word2idx(train_nolabel)  # 无标签数据预处理

    # 数据集，初始为有标签数据，后面后加入无标签数据
    train_x_set, train_y_set = train_x_label, train_y_label

    # 建立模型
    model = GRU(embedding, hidden_dim, num_layers, dropout)
    model = model.to(device)

    for i in range(total_epochs):
        # 先用有标签数据训练模型
        # 划分数据集
        X_train, X_val, y_train, y_val = train_test_split(train_x_set, train_y_set, test_size=0.2, random_state=0)  # 有标签数据集划分一部分作为验证集
        # 把 data 做成 dataset， 供 dataloader 取用
        train_set = TwitterDataset(X_train, y_train)
        val_set = TwitterDataset(X_val, y_val)
        train_loader = DataLoader(train_set, batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size, shuffle=False)
        # 开始训练
        print("{}th :training...".format(i))
        print("训练集大小：{}，测试集大小：{}".format(len(X_train), len(X_val)))
        training(model, device, train_loader, val_loader, batch_size, train_epochs, lr, model_path, best_acc)

        # 预测无标签数据的lable
        test_dataset = TwitterDataset(X=train_x_nolabel, y=None)
        test_loader = DataLoader(test_dataset, batch_size, shuffle=False)
        label_pred, prob_pred = testing(model, device, test_loader)

        # 从无标签数据中选出一部分加入训练集
        # 挑选标准：预测概率大于0.7（正类），小于0.3（负类）
        label_pred, prob_pred = label_pred.view(-1), prob_pred.view(-1)
        index_true, index_false = (prob_pred > 0.7).nonzero(as_tuple=False), (prob_pred < 0.3).nonzero(
            as_tuple=False)  # 分别获取2类索引
        index = torch.cat([index_true, index_false], dim=0).view(-1).to('cpu')  # 2个index拼接
        extra_x = torch.index_select(train_x_nolabel.to('cpu'), 0, index) # 获得新增的训练数据 x
        extra_y = torch.index_select(label_pred.to('cpu'), 0, index)  # 获得新增的训练数据 y

        # 和有标签数据组成新的训练集合
        train_x_set = torch.cat([train_x_label, extra_x], dim=0)
        train_y_set = torch.cat([train_y_label, extra_y], dim=0)
