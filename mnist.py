import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import torch.optim as optim

from net import Net

EPOCHS = 30
LR_RATE = 0.03
MMTUNM = 0.9
WORKERS = 5
BATCH = 100

# GPUが利用可能かどうかをチェックし、利用可能ならGPUを使用
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 使用モデル
model = Net()

# composeによって複数の変換操作を統合
# データをロード
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, ), (0.5, ))])

trainset = torchvision.datasets.MNIST(root='./data', 
                                        train=True,
                                        download=True,
                                        transform=transform)

trainloader = torch.utils.data.DataLoader(trainset,
                                            batch_size=BATCH,
                                            shuffle=True,
                                            num_workers=WORKERS)

testset = torchvision.datasets.MNIST(root='./data', 
                                        train=False, 
                                        download=True, 
                                        transform=transform)

testloader = torch.utils.data.DataLoader(testset, 
                                            batch_size=BATCH,
                                            shuffle=False, 
                                            num_workers=WORKERS)

classes = tuple(np.linspace(0, 9, 10, dtype=np.uint8))



def train_and_save_model():
    model.to(device)  # ネットワークをGPUに移動

    # define loss function and optimier
    criterion = nn.CrossEntropyLoss() # 損失関数
    optimizer = optim.SGD(model.parameters(),
                          lr = LR_RATE, momentum = MMTUNM, nesterov = True)

    # 学習
    for epoch in range(EPOCHS):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader, 0):
            inputs, labels = inputs.to(device), labels.to(device)  # データをGPUに移動

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 100 == 99:
                print('[{:d}, {:5d}] loss: {:.3f}'.format(epoch+1, i+1, running_loss / 100))
                running_loss = 0.0
    print('Finished Training')

    # モデルを保存
    torch.save(model.state_dict(), './mnist_net.pth')



def test_model():
    # モデルをロード
    model.load_state_dict(torch.load('./mnist_net.pth'))
    model.to(device)

    # test
    correct = 0
    total = 0
    with torch.no_grad():
        for (images, labels) in testloader:
            images, labels = images.to(device), labels.to(device)  # データをGPUに移動
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy: {:.2f} %'.format(100 * float(correct / total)))



if __name__ == '__main__':
    start_time = time.time()
    train_and_save_model()
    #test_model()
    print('elapsed time: {:.3f} [sec]'.format(time.time() - start_time))
