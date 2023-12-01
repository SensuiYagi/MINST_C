
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 畳み込み層
        self.conv1 = nn.Conv2d(1, 32, 3) # 3*3カーネル、32個のフィルタ、26x26x32
        self.conv2 = nn.Conv2d(32, 64, 3) # 3*3カーネル、64個のフィルタ、24x24x64

        # プーリング層、Max Pooling
        self.pool = nn.MaxPool2d(2, 2) # 次元削減24x24x64 -> 12x12x64

        # ドロップアウト、過学習防止
        self.dropout1 = nn.Dropout2d() # 50%off
        self.dropout2 = nn.Dropout2d()

        # 全結合層
        self.fc1 = nn.Linear(12 * 12 * 64, 128) # 特徴128個へ
        self.fc2 = nn.Linear(128, 10) # 128へから10個へ

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout1(x)
        x = x.view(-1, 12 * 12 * 64) # フラット化
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

