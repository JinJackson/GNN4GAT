import torch


t1 = torch.randn((6, 3, 2))
t2 = torch.randn((1, 3, 2))

res = t1 * t2

print(res.shape)