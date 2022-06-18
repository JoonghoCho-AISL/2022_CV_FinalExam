from torch import nn


class SRCNN(nn.Module):
    def __init__(self, num_channels=1):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=9 // 2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=5 // 2)
        self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=5 // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x

class FSRCNN(nn.Module):
    def __init__(self, numchannels=1):
        super(FSRCNN, self).__init__()
        self.conv1 = nn.Conv2d(numchannels, 56, kernel_size=4, padding= 'same')
        self.conv2 = nn.Conv2d(56, 16, kernel_size=1)
        self.conv3 = nn.Conv2d(16, 16, kernel_size=3, padding = 'same')
        self.conv4 = nn.Conv2d(16, 56, kernel_size=1)
        self.conv5 = nn.Conv2d(56, 1, kernel_size=9, padding = 'same')
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.conv5(x)