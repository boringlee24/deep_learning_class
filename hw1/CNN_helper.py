import torch.nn as nn
import torch.nn.functional as F


# define the CNN architecture
class MNIST_Classifier(nn.Module):
    def __init__(self):
        super(MNIST_Classifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, padding=2) # 28 -> 28
        self.pool = nn.MaxPool2d(2, 2) # 28 -> 14
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1) # 14 -> 14
        self.fc1 = nn.Linear(64 * 7 * 7, 128) # each channel is 7x7 after pooling
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # print(x.shape)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    

# define the CNN architecture
class CIFAR10_Classifier(nn.Module):
    def __init__(self, dropout_prob=0.4):
        super(CIFAR10_Classifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5, padding=2) # 32 -> 32
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1) # 32 -> 32
        self.bn2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2) # 32 -> 16
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1) # 16 -> 16
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1) # 16 -> 16
        self.bn4 = nn.BatchNorm2d(256)
        self.pool2 = nn.MaxPool2d(2, 2) # 16 -> 8
        self.fc1 = nn.Linear(256 * 8 * 8, 1024) # each channel is 8x8 after pooling
        self.bn5 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 10)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.dropout(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.bn5(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)
        return x