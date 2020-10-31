import torch
from torch import nn
import torch.optim as optim


def evaluation(outputs, labels):
    # outputs => probability (float)
    # labels => labels
    outputs[outputs >= 0.5] = 1  # 大於等於 0.5 為正面
    outputs[outputs < 0.5] = 0  # 小於 0.5 為負面
    correct = torch.sum(torch.eq(outputs, labels)).item()
    return correct


def training(model, device, train, valid, batch_size, epochs, lr, model_path, best_acc=0):
    loss = nn.BCELoss()  # 损失函数为二分类交叉熵损失函数
    t_batch = len(train)
    v_batch = len(valid)
    optimizer = optim.Adam(model.parameters(), lr=lr)  # 优化器为Adam

    for epoch in range(epochs):
        model.train()
        train_loss, train_acc, val_loss, val_acc = 0, 0, 0, 0
        for i, (x, y) in enumerate(train):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()  # 清空梯度
            pred = model(x)  # 预测

            loss_ = loss(pred.squeeze(), y)  # 计算loss
            loss_.backward()  # 计算梯度
            optimizer.step()  # 更新参数

            train_correct = evaluation(pred.squeeze(), y)
            train_acc += (train_correct / batch_size)
            train_loss += loss_.item()
        print('\n epochs:{} | Train | Loss:{:.5f} Acc: {:.3f}'.format(epoch, train_loss / t_batch,
                                                                      train_acc / t_batch * 100))

        model.eval()
        with torch.no_grad():
            for i, (x, y) in enumerate(valid):
                x, y = x.to(device), y.to(device)
                pred = model(x)  # 预测
                loss_ = loss(pred, y.squeeze())

                val_correct = evaluation(pred, y.squeeze())
                val_acc += (val_correct / batch_size)
                val_loss += loss_.item()
        print(
            "epochs:{} | Valid | Loss:{:.5f} Acc: {:.3f} ".format(epoch, val_loss / v_batch, val_acc / v_batch * 100))

        # 保留测试集精度最好的模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model, "{}/ckpt_{}.model".format(model_path, best_acc / v_batch * 100))
            print('saving model with acc {:.3f}'.format(best_acc / v_batch * 100))


def testing(model, device, test):
    model.eval()
    test_pred, prob_pred = [], []
    with torch.no_grad():
        for i, x in enumerate(test):
            x = x.to(device)
            pred = model(x)
            # 预测概率大于等于0.5的为正类
            pred[pred >= 0.5] = 1
            pred[pred < 0.5] = 0
            test_pred += pred.int().tolist()
            prob_pred += pred.tolist()
    return torch.Tensor(test_pred), torch.Tensor(prob_pred)
