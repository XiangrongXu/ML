from linear_util import Timer
from linear_util import *
import linear_util
import torch
from d2l import torch as d2l
import numpy as np
from torch import nn

# n = 10_000
# a = torch.ones(n)
# b = torch.ones(n)
# c = torch.zeros(n)
# timer = Timer()
# for i in range(n):
#     c[i] = a[i] + b[i]
# print(f"{timer.stop():.10f} sec")
# timer.start()
# d = a + b
# print(f"{timer.stop():.10f} sec")


# x = np.arange(-7, 7, 0.01)
# params = [(0, 1), (0, 2), (3, 1)]
# d2l.plot(x, [linear_util.normal(x, mu, sigma) for mu, sigma in params], xlabel="x", ylabel="p(x)", figsize=(4.5, 2.5), legend=[f"mean {mu}, std {sigma}" for mu, sigma in params])
# d2l.plt.show()

# true_w = torch.tensor([2, -3.4])
# true_b = torch.tensor(4.2)
# features, labels = synthetic_data(true_w, true_b, 1_000)
# d2l.set_figsize()
# fig, ax = d2l.plt.subplots(2, 1)
# print(ax.shape)
# ax[0].scatter(features[:, 1].detach().numpy(), labels.detach().numpy(), 1)
# ax[1].scatter(features[:, 1].numpy(), labels.numpy(), 1)
# d2l.plt.show()

net = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 256), nn.ReLU(), nn.Linear(256, 10))
trainer = torch.optim.SGD([{"params": net[1].weight, "weight_decay": 2}, {"params": net[1].bias}, {"params": net[3].weight, "weight_decay": 2}, {"params": net[3].bias}], lr=0.3)
d2l.train_ch3