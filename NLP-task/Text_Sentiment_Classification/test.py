import torch

x = torch.Tensor(5,1)
print(x.shape)
x = x.view(-1)
print(x.shape)
