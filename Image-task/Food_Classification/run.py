from create_model import Resnet
from data_progress import read_file, ImageDataset
from training_testing import training, testing
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Resnet().to(device)

# 读入数据
print('loading data...')
train_x, train_y = read_file('food-11/train1')
val_x, val_y = read_file('food-11/valid1')
test_x = read_file('food-11/test1', lable=False)

# training 时做 data augmentation
print("create dataset...")
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),  # 隨機將圖片水平翻轉
    transforms.RandomRotation(15),  # 隨機旋轉圖片
    transforms.ToTensor(),  # 將圖片轉成 Tensor，並把數值 normalize 到 [0,1] (data normalization)
])
# testing 时不需 data augmentation
test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
])

# 制作数据集
train_set = ImageDataset(train_transform, train_x, train_y)
val_set = ImageDataset(test_transform, val_x, val_y)
test_set = ImageDataset(test_transform, test_x)

train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
val_loader = DataLoader(val_set, batch_size=128, shuffle=False)
test_loader = DataLoader(test_set, batch_size=128, shuffle=False)

# 开始训练
print("training start...")
training(model, device, train_loader, val_loader)
# 测试
print("testing start...")
testing(model, device, test_loader)
