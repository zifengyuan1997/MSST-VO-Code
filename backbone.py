import torch
import torch.nn as nn
import torchvision.models as models
from params import par
from convunit import *

class VGG_BackBone(nn.Module):
    def __init__(self, batchNorm=True, activeFunc = par.activeFunc):
        super(VGG_BackBone, self).__init__()
        self.batchNorm = batchNorm
        self.activeFunc = activeFunc
        # self.conv1 = ConvUnit(self.batchNorm,    3,  16, kernel_size=7, stride=2, dropout=0.1, activeFunc = activeFunc)
        # self.conv2 = ConvUnit(self.batchNorm,   16,  32, kernel_size=5, stride=2, dropout=0.1, activeFunc = activeFunc)
        # self.conv3 = ConvUnit(self.batchNorm,   32,  64, kernel_size=5, stride=2, dropout=0.1, activeFunc = activeFunc)
        # self.conv4 = ConvUnit(self.batchNorm,   64,  64, kernel_size=3, stride=1, dropout=0.1, activeFunc = activeFunc)
        # self.conv5 = ConvUnit(self.batchNorm,   64, 128, kernel_size=3, stride=2, dropout=0.1, activeFunc = activeFunc)
        # self.conv6 = ConvUnit(self.batchNorm,  128, 128, kernel_size=3, stride=1, dropout=0.1, activeFunc = activeFunc)
        # self.conv7 = ConvUnit(self.batchNorm,  128, 128, kernel_size=3, stride=2, dropout=0.1, activeFunc = activeFunc)
        # self.conv8 = ConvUnit(self.batchNorm,  128, 256, kernel_size=3, stride=1, dropout=0.1, activeFunc = activeFunc)
        self.conv1 = ConvUnit(self.batchNorm, 3, 16, kernel_size=7, stride=2, dropout=0.1, activeFunc=activeFunc)
        self.conv2 = ConvUnit(self.batchNorm, 16, 32, kernel_size=5, stride=2, dropout=0.1, activeFunc=activeFunc)
        self.conv3 = ConvUnit(self.batchNorm, 32, 64, kernel_size=5, stride=2, dropout=0.1, activeFunc=activeFunc)
        self.conv4 = ConvUnit(self.batchNorm, 64, 64, kernel_size=3, stride=1, dropout=0.1, activeFunc=activeFunc)
        self.conv5 = ConvUnit(self.batchNorm, 64, 128, kernel_size=3, stride=2, dropout=0.1, activeFunc=activeFunc)
        self.conv6 = ConvUnit(self.batchNorm, 128, 128, kernel_size=3, stride=1, dropout=0.1, activeFunc=activeFunc)
        self.conv7 = ConvUnit(self.batchNorm, 128, 128, kernel_size=3, stride=2, dropout=0.1, activeFunc=activeFunc)
        self.conv8 = ConvUnit(self.batchNorm, 128, 256, kernel_size=3, stride=1, dropout=0.1, activeFunc=activeFunc)
        # for module in self.modules():
        #     print(f"Module: {module}")
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        x7 = self.conv7(x6)
        x8 = self.conv8(x7)
        return x1, x2, x4, x6, x8

class DownStream(nn.Module):
    def __init__(self, batchNorm=True, activeFunc = par.activeFunc, inc = par.inc_down):
        super(DownStream, self).__init__()
        self.batchNorm = batchNorm
        self.activeFunc = activeFunc
        self.inc = inc
        self.conv_down1 = ConvUnit(self.batchNorm, self.inc[0], self.inc[0]*2, kernel_size=1, stride=1,
                                   dropout=0.1, activeFunc=activeFunc)
        self.maxpool_down = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=True)
        self.conv_down2 = ConvUnit(self.batchNorm, self.inc[1], self.inc[1]*2, kernel_size=1, stride=1,
                                   dropout=0.1, activeFunc=activeFunc)
        self.conv_down3 = ConvUnit(self.batchNorm, self.inc[2], self.inc[2]*2, kernel_size=1, stride=1,
                                   dropout=0.1, activeFunc=activeFunc)
        self.conv_down4 = ConvUnit(self.batchNorm, self.inc[3], self.inc[3]*2, kernel_size=1, stride=1,
                                   dropout=0.1, activeFunc=activeFunc)
        self.conv_down5 = ConvUnit(self.batchNorm, self.inc[4], self.inc[4]*2, kernel_size=1, stride=1,
                                   dropout=0.1, activeFunc=activeFunc)
    def forward(self, x1, x2, x3, x4, x5):
        out_down1 = self.maxpool_down(self.conv_down1(x1))
        out_down2 = self.maxpool_down(self.conv_down2(out_down1 + x2))
        out_down3 = self.maxpool_down(self.conv_down3(out_down2 + x3))
        out_down4 = self.maxpool_down(self.conv_down4(out_down3 + x4))
        out_down5 = self.maxpool_down(self.conv_down5(out_down4 + x5)) # torch.Size([4, 1024, 20, 6])
        return out_down5

# class ConvDDD(nn.Module):
#     def __init__(self, batchNorm=True, activeFunc = par.activeFunc):
#         super(ConvDDD, self).__init__()
#         self.batchNorm = batchNorm
#         self.activeFunc = activeFunc
#         self.conv = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
#     def forward(self, x1, x2, x3, x4, x5):
#         x1 = self.globalAverMaxPool(x1)
#         x2 = self.globalAverMaxPool(x2)
#         x3 = self.globalAverMaxPool(x3)
#         x4 = self.globalAverMaxPool(x4)
#         x5 = self.globalAverMaxPool(x5)
#         x1 = conv3ddd(x1)
#         return
#     def globalAverMaxPool(self, x):
#         avgout = torch.mean(x, dim=2, keepdim=True)
#         maxout, _ = torch.max(x, dim=2, keepdim=True)
#         out = torch.cat([avgout, maxout], dim=2)
#         return out

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(8, 3, 1226, 370).to(device)
    Model = VGG_BackBone()
    Model=Model.to(device)
    y = Model(x)
    print(y[0].shape)


