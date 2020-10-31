import torch
import torch.nn as nn
import numpy as np
import time




def training(model, device, train_loader, val_loader):
    # 一些参数设置
    epochs = 30  # 训练伦次
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Adam优化器
    loss = nn.CrossEntropyLoss()  # 交叉熵损失函数

    for epoch in range(epochs):
        epoch_start_time = time.time()
        train_acc, val_acc = 0.0, 0.0
        train_loss, val_loss = 0.0, 0.0

        model.train()  # 训练模式
        train_len = 0
        for i, data in enumerate(train_loader):
            train_len += len(data)
            data = data.to(device)
            optimizer.zero_grad()  # 将模型的梯度清0
            train_pred = model(data[0])  # 模型预测值
            batch_loss = loss(train_pred, data[1])  # 计算损失
            batch_loss.backward()  # 计算每个参数的梯度
            optimizer.step()  # 更新参数

            train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
            train_loss += batch_loss.item()

        model.eval()  # 测试模式
        val_len = 0
        with torch.no_grad():
            for i, data in val_loader:
                val_len += len(data)
                data = data.to(device)
                val_pred = model(data[0])
                batch_loss = loss(val_pred, data[1])

                train_acc += np.sum(np.argmax(val_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
                train_loss += batch_loss.item()
        print('[%03d/%03d] %2.2f sec(s) ,Train Acc: %3.6f, Loss: %3.6f | Val Acc: %3.6f, loss: %3.6f'.format(epoch + 1,
                                                                                                          epochs,
                                                                                                          time.time() - epoch_start_time,
                                                                                                          train_acc / train_len,
                                                                                                          train_loss / train_len,
                                                                                                          val_acc / val_len,
                                                                                                          val_loss / val_len))

def testing(model, device, test_loader):
    model.eval()  # 测试模式
    test_pred = []
    with torch.no_grad():
        for i, data in test_loader:
            data = data.to(device)
            pred = model(data)
            test_label = np.argmax(pred.cpu().data.numpy(), axis=1)
            for lable in test_label:
                test_pred.append(lable)

    # 将预测结果写入文档
    with open("predict.csv", 'w') as f:
        f.write('Id,Category\n')
        for i, y in enumerate(test_pred):
            f.write('{},{}\n'.format(i, y))

