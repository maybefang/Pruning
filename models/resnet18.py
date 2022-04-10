import torch
from torch import nn
from torch.nn import functional as F


class ResBlk(nn.Module):
   #resnet block subblock
    def __init__(self, ch_in, ch_out, stride=1):

        super(ResBlk, self).__init__()

        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)

        self.extra = nn.Sequential()
        #如果输入和输出的通道不一致，或其步长不为 1，需要将二者转成一致
        if ch_out != ch_in:
            # [b, ch_in, h, w] => [b, ch_out, h, w]
            self.extra = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=stride),
                nn.BatchNorm2d(ch_out)
            )

    def forward(self, x):

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out = self.extra(x) + out
        out = F.relu(out)
        return out


class ResNet18(nn.Module):
    '''
      main block
    '''
    def __init__(self, dataset='cifar10', depth=18):
        super(ResNet18, self).__init__()
        if dataset == "cifar10":
            cls = 10
        elif dataset == "cifar100":
            cls = 100

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=3, padding=0),
            nn.BatchNorm2d(64)
        )
        # followed 4 blocks
        self.blk1 = ResBlk(64, 128, stride=2)          #[b, 64, h, w] => [b, 128, h ,w]
        self.blk2 = ResBlk(128, 256, stride=2)         #[b, 128, h, w] => [b, 256, h, w]
        self.blk3 = ResBlk(256, 512, stride=2)         #[b, 256, h, w] => [b, 512, h, w]
        self.blk4 = ResBlk(512, 512, stride=2)         #[b, 512, h, w] => [b, 512, h, w]

        self.outlayer = nn.Linear(512*1*1, cls)         #全连接层，总共10个分类

    def forward(self, x):
        x = F.relu(self.conv1(x))

        # [b, 64, h, w] => [b, 1024, h, w]
        x = self.blk1(x)
        x = self.blk2(x)
        x = self.blk3(x)
        x = self.blk4(x)

        x = F.adaptive_avg_pool2d(x, [1, 1])          #[b, 512, h, w] => [b, 512, 1, 1]
        x = x.view(x.size(0), -1)
        x = self.outlayer(x)

        return x