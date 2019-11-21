import mxnet as mx
from mxnet import autograd
from mxnet import init
from mxnet.gluon import nn
from mxnet import gluon
import mxnet.ndarray as nd
import time
import numpy as np
import csv

# Parameters
EPOCH_NUM = 20    # Training Epoch
BATCH_SIZE = 64   # Batch Size
LR = 0.01         # Learning Rate


# device (CPU or GPU)
# ctx = mx.cpu()
ctx = mx.gpu()


# storing evaluation
class History:
    def __init__(self):
        self.times = []
        self.loss = []
        self.acc = []


# net definition
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


# method of pre-processing data
def transformer(data, label):
    return nd.transpose(data.astype(np.float32), (2, 0, 1)).asnumpy() / 255, label.astype(np.int32)


# training data set
train_set = gluon.data.vision.MNIST(
    train=True,
    transform=transformer,
)

# training data set loader
train_loader = gluon.data.DataLoader(
    dataset=train_set,
    batch_size=BATCH_SIZE,
    shuffle=True
)

# testing data set
test_set = gluon.data.vision.MNIST(
    train=False,
    transform=transformer,
)

# testing data set loader
test_loader = gluon.data.DataLoader(
    dataset=test_set,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# init net
net.initialize(ctx=ctx, init=init.Xavier())
# loss function
criterion = gluon.loss.SoftmaxCrossEntropyLoss()
# optimizer
trainer = gluon.Trainer(
    params=net.collect_params(),
    optimizer='sgd',
    optimizer_params={'learning_rate': LR, 'momentum': 0.9},
)

# Training
if __name__ == "__main__":

    history = History()

    for epoch in range(EPOCH_NUM):
        # one epoch
        epoch_start_time = time.time()
        sum_loss = 0.0

        for X, y in train_loader:
            X, y = X.as_in_context(ctx), y.as_in_context(ctx)
            with autograd.record():
                output = net(X)
                loss = criterion(output, y)
            loss.backward()
            trainer.step(BATCH_SIZE)
            sum_loss += loss.mean().asscalar()

        epoch_time = time.time() - epoch_start_time

        # evaluate
        test_acc = nd.array([0.0], ctx=ctx)
        test_acc = 0
        total = 0
        for X, y in test_loader:
            X, y = X.as_in_context(ctx), y.as_in_context(ctx)
            output = net(X)
            predicted = output.argmax(axis=1)
            test_acc += (predicted == y.astype(np.float32)).sum()
            total += y.size

        epoch_acc = test_acc.asscalar() / total
        epoch_loss = sum_loss / len(train_loader)

        print('epoch: %d, train loss: %.03f, test acc: %.03f, time %.1f sec' % (
        epoch + 1, epoch_loss, epoch_acc, epoch_time))
        history.times.append(epoch_time)
        history.acc.append(epoch_acc)
        history.loss.append(epoch_loss)

    # save result of evaluation
    with open("mxnet-result.csv", "w", newline="") as f:
        f_csv = csv.writer(f)
        headers = ["time", "loss", "acc"]
        f_csv.writerow(headers)
        for i in range(len(history.times)):
            row = [history.times[i], history.loss[i], history.acc[i]]
            f_csv.writerow(row)



