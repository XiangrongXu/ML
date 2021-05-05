import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l
from IPython import display


class Accumulator:
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class Animator:
    """在动画中绘制数据。"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        # 增量地绘制多条线
        if legend is None:
            legend = []
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes,]
        # 使用lambda函数捕获参数
        self.config_axes = lambda: d2l.set_axes(self.axes[
            0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # 向图表中添加多个数据点
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)


class Softmax_Util:
    def __init__(self, W, b):
        self.W = W
        self.b = b

    def get_mnist_datasets(self, transform):
        mnist_train = torchvision.datasets.FashionMNIST(root="./data", train=True, transform=transform, download=True)
        mnist_test = torchvision.datasets.FashionMNIST(root="./data", train=False, transform=transform, download=True)
        return mnist_train, mnist_test

    def get_fashion_mnist_labels(self, labels):
        """返回标签的文字版本"""
        text_labels = ["t-shirst", "trouser", "pullover", "dress", "coat", "sandal", "shirt", "sneaker", "bag", "ankle boot"]
        return [text_labels[i] for i in labels]

    def show_images(self, imgs, num_rows, num_cols, titles=None, scale=1.5):
        figsize = (num_rows* scale, num_cols * scale)
        _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
        axes = axes.flatten()
        for i, (ax, img) in enumerate(zip(axes, imgs)):
            if torch.is_tensor(img):
                ax.imshow(img.numpy())
            else:
                ax.imshow(img)
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            if titles:
                ax.set_title(titles[i])
        return axes

    def get_num_workers(self):
        return 4

    def load_data_fashion_mnist(self, batch_size, resize=None):
        trans = [transforms.ToTensor()]
        if resize:
            trans.insert(0, transforms.Resize(resize))
        trans = transforms.Compose(trans)
        mnist_train, mnist_test = self.get_mnist_datasets(trans)
        return data.DataLoader(mnist_train, batch_size, shuffle=True), data.DataLoader(mnist_test, batch_size)

    def softmax(self, X):
        X_exp = torch.exp(X)
        partition = X_exp.sum(dim=1, keepdim=True)
        return X_exp / partition

    def net(self, X):
        print(X.shape)
        return self.softmax(torch.matmul(X.reshape(-1, len(self.W)), self.W) + self.b)

    def cross_entropy(self, y_hat, y):
        return -torch.log(y_hat[range(len(y_hat)), y])

    def accuracy(self, y_hat, y):
        y_hat = y_hat.argmax(axis=1)
        cmp = y_hat.type(y.dtype) == y
        return float(cmp.type(y.dtype).sum())

    def evaluate_accuracy(self, net, data_iter):
        if isinstance(net, torch.nn.Module):
            net.eval()
        metric = Accumulator(2)
        for X, y in data_iter:
            metric.add(self.accuracy(net(X), y), y.numel())
        return metric[0] / metric[1]

    def train_epoch_ch3(self, net, train_iter, loss, updater):
        if isinstance(net, torch.nn.Module):
            net.train()
        metric = Accumulator(3)
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y)
            if isinstance(updater, torch.optim.Optimizer):
                updater.zero_grad()
                l.backward()
                updater.step()
                metric.add(float(l) * len(y), self.accuracy(y_hat, y), y.numel())
            else:
                l.sum().backward()
                updater(X.shape[0])
                metric.add(float(l.sum()), self.accuracy(y_hat, y), y.numel())

        return metric[0] / metric[2], metric[1] / metric[2]

    def train_ch3(self, net, train_iter, test_iter, loss, num_epochs, updater):
        animator = Animator(xlabel="epoch", xlim=[1, num_epochs], ylim=[0.3, 0.9], legend=["train_loss, train_acc", "test_acc"])
        for epoch in range(num_epochs):
            train_metrics = self.train_epoch_ch3(net, train_iter, loss, updater)
            test_acc = self.evaluate_accuracy(net, test_iter)
            animator.add(epoch + 1, train_metrics + (test_acc, ))
        train_loss, train_acc = train_metrics
        # assert train_loss < 0.5, train_loss
        # assert train_acc <= 1 and train_acc > 0.7, train_acc
        # assert test_acc <= 1 and test_acc > 0.7, test_acc

    def sgd(self, params, lr, batch_size):
        with torch.no_grad():
            for param in params:
                param -= lr * param.grad / batch_size
                param.grad.zero_()

    def updater(self, batch_size):
        return self.sgd([self.W, self.b], 0.03, batch_size)

    def predict_ch3(self, net, test_iter, n=6):
        for X, y in test_iter:
            break
        trues = self.get_fashion_mnist_labels(y)
        preds = self.get_fashion_mnist_labels(net(X).argmax(axis=1))
        titles = [true + "\n" + pred for true, pred in zip(trues, preds)]
        self.show_images(X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])
        d2l.plt.show()



# d2l.use_svg_display()
# mnist_train, _ = get_mnist_datasets()
# data_iter = data.DataLoader(mnist_train, batch_size=18)
# X, y = next(iter(data_iter))
# show_images(X.reshape(18, 28, 28), 2, 9, titles=get_fashion_mnist_labels(y))
# d2l.plt.show()
