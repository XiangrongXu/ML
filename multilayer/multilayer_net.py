import torch
from torch import nn
from d2l import torch as d2l

class A:
    def __init__(self):
        self.data = [0.0, 0.0]

    def evaluate_accuracy(self, y_hat, y):
        y_hat = y_hat.argmax(axis=1)
        cmp = y_hat.type(y.dtype) == y
        self.data = [a + b for a, b in zip(self.data, (cmp.sum(), y.numel()))]

    def __getitem__(self, idx):
        return self.data[idx]

net = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 256), nn.ReLU(), nn.Linear(256, 10), nn.Softmax())
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)
net.apply(init_weights)
batch_size, lr, num_epochs = 256, 0.1, 10
loss = nn.CrossEntropyLoss()
trainer = torch.optim.SGD(net.parameters(), lr=lr)
for epoch in range(num_epochs):
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    a = A()
    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    for X, y in test_iter:
        a.evaluate_accuracy(net(X), y)
    print(f"acc: {a[0] / a[1]}")
    