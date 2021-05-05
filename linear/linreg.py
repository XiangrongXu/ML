import torch
import random
import linear_util


# generate a handle of items of synthetic data
true_w = torch.tensor([2, -3.4])
true_b = torch.tensor(4.2)
num_examples = 1_000
features, labels = linear_util.synthetic_data(true_w, true_b, num_examples)

w = torch.normal(0, 1, (2,), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
batch_size = 10
num_epochs = 3
lr = 0.03
loss = linear_util.squared_loss
net = linear_util.linreg
for epoch in range(num_epochs):
    for X, y in linear_util.data_iter(features, labels, batch_size):
        l = loss(net(X, w, b), y)
        l.sum().backward()
        linear_util.sgd([w, b], lr, batch_size)
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f"epoch: {epoch + 1}, loss: {train_l.mean():f}")
        print(f"w: {w}, b: {b}")
        print("=" * 15)
