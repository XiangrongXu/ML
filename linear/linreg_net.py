import torch
from torch import nn
from torch.utils import data
import linear_util


true_w = torch.tensor([2, -3.4])
true_b = torch.tensor(4.2)
num_examples = 1_000
features, labels = linear_util.synthetic_data(true_w, true_b, num_examples)

batch_size = 10
data_iter = linear_util.load_array((features, labels), batch_size)
net = nn.Sequential(nn.Linear(2, 1))
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)
loss = nn.MSELoss()
trainer = torch.optim.SGD(net.parameters(), lr=0.03)

num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X), y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    train_l = loss(net(features), labels)
    print(f"epoch: {epoch + 1}, loss: {train_l:f}")

print(net[0].weight.data, net[0].bias.data)