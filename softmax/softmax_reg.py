import torch
from softmax_util import Softmax_Util

num_inputs = 28 * 28
num_outputs = 10
W = torch.normal(0, 0.01, (num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)
util = Softmax_Util(W, b)
batch_size = 256
train_iter, test_iter = util.load_data_fashion_mnist(batch_size)
num_epochs = 10
net = util.net
loss = util.cross_entropy
updater = util.updater
util.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)

util.predict_ch3(net, test_iter)





