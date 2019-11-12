import mxnet as mx
from mxnet import autograd
from mxnet import init
from mxnet.gluon import nn
from mxnet import gluon
import mxnet.ndarray as nd
import time
import numpy as np
import csv

class History:
    def __init__(self):
        self.times = []
        self.loss = []
        self.acc = []


net = nn.Sequential()
net.add(
    nn.Conv2D(channels=6, kernel_size=5, strides=1, padding=2, activation='relu'),
    nn.MaxPool2D(pool_size=2, strides=2),
    nn.Conv2D(channels=16, kernel_size=5, activation='relu'),
    nn.MaxPool2D(pool_size=2, strides=2),
    nn.Flatten(),
    nn.Dense(120, activation='relu'),
    nn.Dense(84, activation='relu'),
    nn.Dense(10),
)

epoch_nums = 2
batch_size = 128
lr = 0.01

def transformer(data, label):
    return nd.transpose(data.astype(np.float32), (2, 0, 1)).asnumpy() / 255, label.astype(np.int32)


train_set = gluon.data.vision.MNIST(
    train=True,
    transform=transformer,
)

train_loader = gluon.data.DataLoader(
    dataset=train_set,
    batch_size=batch_size,
    shuffle=True
)

test_set = gluon.data.vision.MNIST(
    train=False,
    transform=transformer,
)

test_loader = gluon.data.DataLoader(
    dataset=test_set,
    batch_size=batch_size,
    shuffle=False
)

ctx = mx.cpu()
net.initialize(ctx=ctx, init=init.Xavier())
criterion = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(
    params=net.collect_params(),
    optimizer='sgd',
    optimizer_params={'learning_rate': lr, 'momentum': 0.9},
)

history = History()
for epoch in range(epoch_nums):
    loss_sum = 0
    start_time = time.time()
    for X, y in train_loader:
        X, y = X.as_in_context(ctx), y.as_in_context(ctx)
        with autograd.record():
            output = net(X)
            loss = criterion(output, y)
        loss.backward()
        trainer.step(batch_size)
        loss_sum += loss.mean().asscalar()

    test_acc = nd.array([0.0], ctx=ctx)
    test_acc = 0
    total = 0
    for X, y in test_loader:
        X, y = X.as_in_context(ctx), y.as_in_context(ctx)
        output = net(X)
        predicted = output.argmax(axis=1)
        test_acc += (predicted == y.astype(np.float32)).sum()
        total += y.size
    epoch_time = time.time() - start_time
    epoch_acc = test_acc.asscalar() / total
    epoch_loss = loss_sum / len(train_loader)

    print('epoch: %d, train loss: %.03f, test acc: %.03f, time %.1f sec' % (
    epoch + 1, epoch_loss, epoch_acc, epoch_time))
    history.times.append(epoch_time)
    history.acc.append(epoch_acc)
    history.loss.append(epoch_loss)

with open("record.csv", "w", newline="") as f:
    f_csv = csv.writer(f)
    headers = ["time", "loss", "acc"]
    f_csv.writerow(headers)
    for i in range(len(history.times)):
        row = [history.times[i], history.loss[i], history.acc[i]]
        f_csv.writerow(row)



