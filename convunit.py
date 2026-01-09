import torch
import torch.nn as nn




class ConvUnit1_1(nn.Module):
    def __init__(self):
        super(ConvUnit1_1, self).__init__()
        # self.conv_x1 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(1, 1), stride=1, padding=0)
        # self.conv_x2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(1, 1), stride=1, padding=0)
        # self.conv_x3 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(1, 1), stride=1, padding=0)
        # self.conv_x4 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(1, 1), stride=1, padding=0)
        # self.conv_x5 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(1, 1), stride=1, padding=0)
        # self.bn1 = nn.BatchNorm2d(16)
        # self.bn2 = nn.BatchNorm2d(32)
        # self.bn3 = nn.BatchNorm2d(64)
        # self.bn4 = nn.BatchNorm2d(128)
        # self.bn5 = nn.BatchNorm2d(256)
        self.conv_x1 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(1, 1), stride=1, padding=0)
        self.conv_x2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(1, 1), stride=1, padding=0)
        self.conv_x3 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(1, 1), stride=1, padding=0)
        self.conv_x4 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(1, 1), stride=1, padding=0)
        self.conv_x5 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(1, 1), stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(256)
        self.RELU = nn.LeakyReLU(0.1)
    def forward(self, x1, x2, x3, x4, x5):
        x1 = self.RELU(self.bn1(self.conv_x1(x1)))
        x2 = self.RELU(self.bn2(self.conv_x2(x2)))
        x3 = self.RELU(self.bn3(self.conv_x3(x3)))
        x4 = self.RELU(self.bn4(self.conv_x4(x4)))
        x5 = self.RELU(self.bn5(self.conv_x5(x5)))
        return x1, x2, x3, x4, x5

def ConvUnit(batchNorm, in_planes, out_planes, kernel_size=3, stride=1, dropout=0, activeFunc = 'lrelu'):
    if activeFunc == 'lrelu':
        if batchNorm:
          return nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=False),
                nn.BatchNorm2d(out_planes),
                nn.LeakyReLU(0.1),
                nn.Dropout(dropout)#, inplace=True)
            )
        else:
            return nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True),
                nn.LeakyReLU(0.1),
                nn.Dropout(dropout)#, inplace=True)
            )
    elif activeFunc == 'relu':
        if batchNorm:
            return nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2,
                          bias=False),
                nn.BatchNorm2d(out_planes),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)  # , inplace=True)
            )
        else:
            return nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2,
                          bias=True),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)  # , inplace=True)
            )

def ConvUnit3D(batchNorm, in_planes, out_planes, kernel_size=(3,3,3), stride=(1,1,1), dropout=0, activeFunc='lrelu'):
    if activeFunc == 'lrelu':
        if batchNorm:
          return nn.Sequential(
                nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                          padding = ((kernel_size[0]-1)//2, (kernel_size[1]-1)//2, (kernel_size[2]-1)//2), bias=False),
                nn.BatchNorm3d(out_planes),
                nn.LeakyReLU(0.1),
                nn.Dropout3d(dropout)#, inplace=True)
            )
        else:
            return nn.Sequential(
                nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                          padding=((kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2, (kernel_size[2] - 1) // 2),
                          bias=False),
                nn.LeakyReLU(0.1),
                nn.Dropout3d(dropout)#, inplace=True)
            )
    elif activeFunc == 'relu':
        if batchNorm:
          return nn.Sequential(
                nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                          padding = ((kernel_size[0]-1)//2, (kernel_size[1]-1)//2, (kernel_size[2]-1)//2), bias=False),
                nn.BatchNorm3d(out_planes),
                nn.ReLU(inplace=True),
                nn.Dropout3d(dropout)#, inplace=True)
            )
        else:
            return nn.Sequential(
                nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                          padding=((kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2, (kernel_size[2] - 1) // 2),
                          bias=False),
                nn.ReLU(inplace=True),
                nn.Dropout3d(dropout)#, inplace=True)
            )
