import torch
import random

def synthetic_data(true_w, true_b, num_examples):
    """
    @param true_w: torch.tensor             the true parameter tensor of the weights to the model
    @param true_b: torch.tensor or int      the true parameter tensor of the biases to the model
    @param num_examples: int                the number of the data items
    @return X, y                            X is the tensor of synthenic features; y is the tensor of synthenic labels
    """
    X = torch.normal(0, 1, (num_examples, len(true_w)))
    y = torch.matmul(X, true_w) + true_b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))


def data_iter(X, y, batch_size):
    """
    @param X: torch.tensor      the tensor of features
    @param y: torch.tensor      the tensor of labels
    @param batch_size: int      the batch_size
    @yield sub_X, sub_y         select a batch from X and y
    """
    num_examples = len(X)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        subscripts = indices[i:min(i + batch_size, num_examples)]
        yield X[subscripts], y[subscripts]


def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2


def linreg(X, w, b):
    return torch.matmul(X, w) + b


def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


# generate a handle of items of synthetic data
true_w = torch.tensor([2, -3.4])
true_b = torch.tensor(4.2)
num_examples = 1_000
features, labels = synthetic_data(true_w, true_b, num_examples)

w = torch.normal(0, 1, (2,), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
batch_size = 10
num_epochs = 3
lr = 0.03
loss = squared_loss
net = linreg
for epoch in range(num_epochs):
    for X, y in data_iter(features, labels, batch_size):
        l = loss(net(X, w, b), y)
        l.sum().backward()
        sgd([w, b], lr, batch_size)
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f"epoch: {epoch + 1}, loss: {train_l.mean():f}")
        print(f"w: {w}, b: {b}")
        print("=" * 15)
