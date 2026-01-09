import torch
import torch.nn as nn
import torchvision.ops as ops


class DeformableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, mode = 2):
        super(DeformableConv2d, self).__init__()

        # 常规卷积参数
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.1)
        self.mode=mode
        self.dilation = dilation
        # 偏移量预测卷积层
        # 输出通道数为2*kernel_size*kernel_size（x和y方向偏移）
        self.offset_conv = nn.Conv2d(in_channels,
                                     2 * self.kernel_size[0] * self.kernel_size[1],
                                     kernel_size=self.kernel_size,
                                     stride=stride,
                                     padding=self.padding, dilation=self.dilation)

        if self.mode == 1:
            # 可变形卷积的权重参数
            self.weight = nn.Parameter(torch.empty(out_channels,
                                                   in_channels,
                                                   self.kernel_size[0],
                                                   self.kernel_size[1]))

            # 偏置参数
            self.bias = nn.Parameter(torch.empty(out_channels))
        if self.mode == 2:
            # 可变形卷积的权重参数
            self.weight = nn.Parameter(torch.empty(out_channels//2,
                                                   in_channels//2,
                                                   self.kernel_size[0],
                                                   self.kernel_size[1]))

            # 偏置参数
            self.bias = nn.Parameter(torch.empty(out_channels//2))

        # 初始化参数
        nn.init.kaiming_uniform_(self.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.bias, 0)
        nn.init.constant_(self.offset_conv.weight, 0)
        # 初始化偏移量卷积的偏置，使初始偏移为0
        self.offset_conv.register_parameter('bias',
                                            nn.Parameter(torch.zeros(2 * self.kernel_size[0] * self.kernel_size[1])))

    def forward(self, x, offset):
        # 1. 预测偏移量
        if offset is None:
            offset = self.offset_conv(x)
        # print(offset.shape, x.shape)

        if self.mode == 1:
            out = ops.deform_conv2d(input=x,
                                     offset=offset,
                                     weight=self.weight,
                                     bias=self.bias,
                                     stride=self.stride,
                                     padding=self.padding,
                                     dilation=(self.dilation, self.dilation))
        if self.mode == 2:
            x1,x2 = torch.tensor_split(x, indices=[x.shape[1]//2], dim=1)
            # print(x1.shape, x2.shape)

            # 2. 应用可变形卷积
            out1 = ops.deform_conv2d(input=x1,
                                    offset=offset,
                                    weight=self.weight,
                                    bias=self.bias,
                                    stride=self.stride,
                                    padding=self.padding,
                                    dilation=(self.dilation, self.dilation))
            out2 = ops.deform_conv2d(input=x2,
                                     offset=offset,
                                     weight=self.weight,
                                     bias=self.bias,
                                     stride=self.stride,
                                     padding=self.padding,
                                     dilation=(self.dilation, self.dilation))
            out = torch.cat([out1, out2], dim=1)

        return out, offset


# 测试代码
if __name__ == "__main__":
    # 创建输入张量 (batch_size=4, channels=3, height=32, width=32)
    x = torch.rand(4, 16, 50, 50)

    # 创建可变形卷积层 (3输入通道 -> 16输出通道, 3x3卷积核)
    deform_conv = DeformableConv2d(in_channels=16, out_channels=16, mode=1)

    # 前向传播
    out = deform_conv(x)

    print("输入形状:", x.shape)
    print("输出形状:", out.shape)  # 应该输出 torch.Size([4, 16, 32, 32])