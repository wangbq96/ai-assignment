import torch
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import time
import csv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Sequential(
            # in_channel, out_channel, kernel_size, stride, padding
            nn.Conv2d(1, 6, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),          # input_size=(16*10*10)
            nn.MaxPool2d(2, 2)  # output_size=(16*5*5)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU()
        )
        self.fc3 = nn.Linear(84, 10)

    def forward(self, input_):
        x = self.conv1(input_)
        x = self.conv2(x)
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

class History:
    def __init__(self):
        self.times = []
        self.loss = []
        self.acc = []


# 超参数设置
EPOCH = 3          # 遍历数据集次数
BATCH_SIZE = 128    # 批处理尺寸(batch_size)
LR = 0.01         # 学习率

# 定义数据预处理方式
transform = transforms.ToTensor()

# 定义训练数据集
train_set = tv.datasets.MNIST(
    root='./data/',
    train=True,
    download=True,
    transform=transform
)

# 定义训练批处理数据
train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

# 定义测试数据集
test_set = tv.datasets.MNIST(
    root='./data/',
    train=False,
    download=True,
    transform=transform
)

# 定义测试批处理数据
test_loader = torch.utils.data.DataLoader(
    test_set,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

# 定义损失函数loss function 和优化方式（采用SGD）
net = LeNet().to(device)
criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数，通常用于多分类问题上
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9)

# 训练
if __name__ == "__main__":

    history = History()

    print("Start training...")
    for epoch in range(EPOCH):
        epoch_time_start = time.time()
        sum_loss = 0.0
        count = 0
        # 数据读取
        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # 梯度清零
            optimizer.zero_grad()

            # forward + backward
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            sum_loss += loss.item()
            count += 1

        epoch_time = time.time()-epoch_time_start
        history.times.append(epoch_time)
        # evaluate
        with torch.no_grad():
            correct = 0
            total = 0
            for data in test_loader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                # 取得分最高的那个类
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()

            epoch_loss = sum_loss / count
            history.loss.append(epoch_loss)
            epoch_acc = correct.item() / total
            history.acc.append(epoch_acc)

        print("time: {}, loss: {}, accuracy: {}".format(epoch_time, epoch_loss, epoch_acc))

    with open("record.csv", "w", newline="") as f:
        f_csv = csv.writer(f)
        headers = ["time", "loss", "acc"]
        f_csv.writerow(headers)
        for i in range(len(history.times)):
            row = [history.times[i], history.loss[i], history.acc[i]]
            f_csv.writerow(row)

