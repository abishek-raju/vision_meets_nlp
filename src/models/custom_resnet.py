import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out




class Custom_ResNet(nn.Module):
    def __init__(self):
        super(Custom_ResNet, self).__init__()
        #Input size 32

        # Prep Layer
        self.preplayer_convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), padding=1, bias=False,
                        stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),

        ) # output_size = 32

        #Layer 1
        self.layer1_convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1, bias=False,
                        stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),
        ) # output_size = 16

        self.layer1_resblock = nn.Sequential(
            BasicBlock(64,128),
            nn.Conv2d(128, 128, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(128),
        )
        # output_size = 16


        #Layer 2
        self.layer2_convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1, bias=False,
                        stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2, 2),
        ) # output_size = 8

        #Layer 3
        self.layer3_convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding=1, bias=False,
                        stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2, 2),
        ) # output_size = 4

        self.layer3_resblock = nn.Sequential(
            BasicBlock(256,512),
            nn.Conv2d(512, 512, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(512),
        )
        # output_size = 4

        self.maxpool_k4 = nn.MaxPool2d(4, 4)

        self.linear = nn.Linear(512,10)

    def forward(self,x):
        #size (n,3,32,32)
        x = self.preplayer_convblock1(x)
        #size (n,64,32,32)

        x = self.layer1_convblock1(x) + self.layer1_resblock(x)
        #size (n,128,16,16) = (n,128,16,16) + (n,128,16,16)

        x = self.layer2_convblock1(x)
        #size (n,256,8,8)

        x = self.layer3_convblock1(x) + self.layer3_resblock(x)
        #size (n,512,4,4) = (n,512,4,4) + (n,512,4,4)

        x = self.maxpool_k4(x)
        #size (n,512,1,1)

        x = x.view(x.size(0), -1)
        #size (n,512)

        x = self.linear(x)
        #size (n,10)
        return F.log_softmax(x, dim=-1)