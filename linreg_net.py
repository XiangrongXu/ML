import torch
from torch import nn
from torch.utils import data

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


def load_array(data_arrays, batch_size, is_train=True):
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


true_w = torch.tensor([2, -3.4])
true_b = torch.tensor(4.2)
num_examples = 1_000
features, labels = synthetic_data(true_w, true_b, num_examples)

batch_size = 10
data_iter = load_array((features, labels), batch_size)
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