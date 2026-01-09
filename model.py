import torch
import torch.nn as nn
from torch.nn.functional import conv2d
from torchvision.ops.deform_conv import *
from params import par
from torch.autograd import Variable
from torch.nn.init import kaiming_normal_, orthogonal_
import numpy as np
from convunit import *
from backbone import *
from convlstm import *
from crossAttention import *
# from cnn import *
# from DeformConv2d import *
from DeformableConv import *

class MSSTVO(nn.Module):
    def __init__(self, datasetused, batchNorm=True, activeFunc=par.activeFunc,
                 backBoneLayer=par.backBoneLayer, seq_len=par.seq_len):
        super(MSSTVO, self).__init__()
        # CNN
        self.deformMode = 2 # 2: proposed, 1: traditional
        self.datasetused = datasetused
        self.batchNorm = batchNorm
        self.activeFunc = activeFunc
        self.backBoneLayer = backBoneLayer
        self.seq_len = seq_len[0] - 1
        self.clip = par.clip
        self.backboneNet = VGG_BackBone()
        self.downNet = DownStream()
        self.conv1_1 = ConvUnit1_1()
        if self.deformMode == 2:
            self.deform1_1 = DeformableConv2d(32, 32, dilation=1, padding=1)
            self.deform2_1 = DeformableConv2d(64, 64, dilation=1, padding=1)
            self.deform3_1 = DeformableConv2d(128, 128, dilation=1, padding=1)
        elif self.deformMode == 1:
            self.deform1_1 = DeformableConv2d(8,8, dilation=1, padding = 1, mode = self.deformMode)
        else:
            raise NotImplementedError

        self.convlstm_x4 = ConvLSTM(input_dim=128, hidden_dim=[128, 128, 128], kernel_size=(3, 3),
                                    dilation=1, num_layers=3, batch_first=True, bias=True, return_all_layers=False)
        self.convlstm_x5 = ConvLSTM(input_dim=256, hidden_dim=[256, 256, 256], kernel_size=(3, 3),
                                    dilation=1, num_layers=3, batch_first=True, bias=True, return_all_layers=False)

        self.conv1_1_x5 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.linear_x5 = nn.Linear(in_features=512, out_features=256)
        self.kesi = nn.Parameter(torch.zeros(1))
        self.kesi2 = nn.Parameter(torch.ones(1))
        self.bnx5ca = nn.BatchNorm2d(256)
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True)

        self.RELU = nn.LeakyReLU(0.1)

        self.crossAtten = MultiheadAttention_v3(256, 8)  # ImageMultiheadCrossAttention(256, 8)

        # Comput the shape based on diff image size
        if self.datasetused == 'KITTI':
            __tmp = Variable(torch.zeros(1, 1, 512, 10, 3))#512
        elif self.datasetused == 'VOD':
            __tmp = Variable(torch.zeros(1, 1, 512, 8, 5))
        else:
            raise ValueError('datasetused must be KITTI or VOD')

        # RNN
        self.rnn = nn.LSTM(
            input_size=int(np.prod(__tmp.size())),
            hidden_size=par.rnn_hidden_size,
            num_layers=2,
            dropout=par.rnn_dropout_between,
            batch_first=True)
        self.rnn_drop_out = nn.Dropout(par.rnn_dropout_out)
        self.linear = nn.Linear(in_features=par.rnn_hidden_size, out_features=6)

        # Initilization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.LSTM):
                # layer 1
                kaiming_normal_(m.weight_ih_l0)  # orthogonal_(m.weight_ih_l0)
                kaiming_normal_(m.weight_hh_l0)
                m.bias_ih_l0.data.zero_()
                m.bias_hh_l0.data.zero_()
                # Set forget gate bias to 1 (remember)
                n = m.bias_hh_l0.size(0)
                start, end = n // 4, n // 2
                m.bias_hh_l0.data[start:end].fill_(1.)

                # layer 2
                kaiming_normal_(m.weight_ih_l1)  # orthogonal_(m.weight_ih_l1)
                kaiming_normal_(m.weight_hh_l1)
                m.bias_ih_l1.data.zero_()
                m.bias_hh_l1.data.zero_()
                n = m.bias_hh_l1.size(0)
                start, end = n // 4, n // 2
                m.bias_hh_l1.data[start:end].fill_(1.)

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        # x: (batch, seq_len, channel, width, height)
        # torch.Size([8, 8, 3, 608, 184])

        batch_size = x.size(0)  #
        seq_lenlen = x.size(1)  # 8

        x1, x2, x3, x4, x5 = self.encode_image(x, seq_lenlen)

        x5 = self.RELU(self.kesi * self.interactAtten(x5)) + x5
        # print(self.interactAtten(x5).shape)
        # print(x5.device)
        xx1 = torch.empty(x1.size(0), x1.size(1) - 1, x1.size(2)*2, x1.size(3), x1.size(4)).cuda()
        xx2 = torch.empty(x2.size(0), x2.size(1) - 1, x2.size(2)*2, x2.size(3), x2.size(4)).cuda()
        xx3 = torch.empty(x3.size(0), x3.size(1) - 1, x3.size(2) * 2, x3.size(3), x3.size(4)).cuda()

        offset1_t = torch.empty(x1.size(0), x1.size(1) - 1, 18, x1.size(3), x1.size(4)).cuda()
        offset2_t = torch.empty(x2.size(0), x2.size(1) - 1, 18, x2.size(3), x2.size(4)).cuda()
        offset3_t = torch.empty(x3.size(0), x3.size(1) - 1, 18, x3.size(3), x3.size(4)).cuda()

        for i in range(seq_lenlen - 1):
            xx1[:, i, :, :, :], offset1 = self.deform1_1(torch.cat((x1[:, i, :, :, :], x1[:, i + 1, :, :, :]), dim=1), None)
            offset1_t[:, i, :, :, :] = offset1

        for i in range(seq_lenlen - 1):
            xx2[:, i, :, :, :], offset2 = self.deform2_1(torch.cat((x2[:, i, :, :, :], x2[:, i + 1, :, :, :]), dim=1), self.avgpool(offset1_t[:, i, :, :, :] / 2))
            offset2_t[:, i, :, :, :] = offset2

        for i in range(seq_lenlen - 1):
            xx3[:, i, :, :, :], offset3 = self.deform3_1(torch.cat((x3[:, i, :, :, :], x3[:, i + 1, :, :, :]), dim=1), self.avgpool(offset2_t[:, i, :, :, :] / 2))

        x1, x2, x3, x4, x5 = self.conv1by1(xx1+torch.cat((x1[:, :-1], x1[:, 1:]), dim=2),#+
                                           xx2+torch.cat((x2[:, :-1], x2[:, 1:]), dim=2),#
                                           xx3+torch.cat((x3[:, :-1], x3[:, 1:]), dim=2),#
                                           torch.cat((x4[:, :-1], x4[:, 1:]), dim=2),
                                           torch.cat((x5[:, :-1], x5[:, 1:]), dim=2), batch_size, seq_lenlen)

        x4 = x4 + self.RELU(self.convlstm_x4(x4)[0][0])
        x5 = x5 + self.RELU(self.convlstm_x5(x5)[0][0])

        if self.datasetused == 'KITTI':
            out = torch.empty(batch_size, seq_lenlen - 1, 512, 10, 3).cuda()#512
        elif self.datasetused == 'VOD':
            out = torch.empty(batch_size, seq_lenlen - 1, 512, 8, 5).cuda()
        else:
            raise ValueError('datasetused must be KITTI or VOD')

        for i in range(seq_lenlen - 1):
            out[:, i, :, :, :] = self.down_stream(torch.squeeze(x1[:, i, :, :, :], dim=1),
                                                  torch.squeeze(x2[:, i, :, :, :], dim=1),
                                                  torch.squeeze(x3[:, i, :, :, :], dim=1),
                                                  torch.squeeze(x4[:, i, :, :, :], dim=1),
                                                  torch.squeeze(x5[:, i, :, :, :], dim=1))
        del x1, x2, x3, x4, x5
        out = out.view(batch_size, seq_lenlen - 1, -1)
        # out = out.cuda()

        # RNN
        out, hc = self.rnn(out)  # torch.Size([8, 7, 1000])
        out = self.rnn_drop_out(out)  # torch.Size([8, 7, 1000])
        out = self.linear(out)  # torch.Size([8, 7, 6])
        return out

    def interactAtten(self, inn):
        # torch.Size([1, 8, 256, 39, 12])
        len = inn.size(1)
        x = torch.empty(inn.size(1), inn.size(3) * inn.size(4), inn.size(0), inn.size(2)).cuda()
        x5 = inn.view(inn.size(0), inn.size(1), inn.size(2), inn.size(3) * inn.size(4)).permute(1, 3, 0, 2)

        for i in range(len):
            if i == 0:
                x[i, :, :, :] = self.crossAtten(x5[i, :, :, :], x5[i + 1, :, :, :], x5[i + 1, :, :, :])
            elif i == len - 1:
                x[i, :, :, :] = self.crossAtten(x5[i, :, :, :], x5[i - 1, :, :, :], x5[i - 1, :, :, :])
            else:
                x[i, :, :, :] = self.crossAtten(x5[i, :, :, :], 
                                                self.linear_x5(torch.cat((x5[i - 1, :, :, :], x5[i + 1, :, :, :]), dim=2)), 
                                                self.linear_x5(torch.cat((x5[i - 1, :, :, :], x5[i + 1, :, :, :]), dim=2)))

        x5 = x5.permute(2, 0, 3, 1).view(inn.size(0), inn.size(1), inn.size(2), inn.size(3), inn.size(4))
        x55 = x5
        for i in range(len):
            x55[:,i,:,:,:] = self.bnx5ca(x5[:,i,:,:,:])

        return x55

    def encode_image(self, x, seq_len):
        batch_size = x.size(0)
        x = x.view(batch_size * seq_len, x.size(2), x.size(3), x.size(4))
        x1, x2, x3, x4, x5 = self.backboneNet(x)
        x1 = x1.view(batch_size, seq_len, x1.size(1), x1.size(2), x1.size(3))
        x2 = x2.view(batch_size, seq_len, x2.size(1), x2.size(2), x2.size(3))
        x3 = x3.view(batch_size, seq_len, x3.size(1), x3.size(2), x3.size(3))
        x4 = x4.view(batch_size, seq_len, x4.size(1), x4.size(2), x4.size(3))
        x5 = x5.view(batch_size, seq_len, x5.size(1), x5.size(2), x5.size(3))
        return x1, x2, x3, x4, x5

    def down_stream(self, x1, x2, x3, x4, x5):
        x = self.downNet(x1, x2, x3, x4, x5)
        return x

    def conv1by1(self, x1, x2, x3, x4, x5, batch_size, seqlen):
        x1 = x1.view(batch_size * (seqlen - 1), x1.size(2), x1.size(3), x1.size(4))
        x2 = x2.view(batch_size * (seqlen - 1), x2.size(2), x2.size(3), x2.size(4))
        x3 = x3.view(batch_size * (seqlen - 1), x3.size(2), x3.size(3), x3.size(4))
        x4 = x4.view(batch_size * (seqlen - 1), x4.size(2), x4.size(3), x4.size(4))
        x5 = x5.view(batch_size * (seqlen - 1), x5.size(2), x5.size(3), x5.size(4))
        x1, x2, x3, x4, x5 = self.conv1_1(x1, x2, x3, x4, x5)
        x1 = x1.view(batch_size, seqlen - 1, x1.size(1), x1.size(2), x1.size(3))
        x2 = x2.view(batch_size, seqlen - 1, x2.size(1), x2.size(2), x2.size(3))
        x3 = x3.view(batch_size, seqlen - 1, x3.size(1), x3.size(2), x3.size(3))
        x4 = x4.view(batch_size, seqlen - 1, x4.size(1), x4.size(2), x4.size(3))
        x5 = x5.view(batch_size, seqlen - 1, x5.size(1), x5.size(2), x5.size(3))
        return x1, x2, x3, x4, x5

    def averMaxPool(self, x):
        avgout = torch.mean(x, dim=2, keepdim=True)
        maxout, _ = torch.max(x, dim=2, keepdim=True)
        out = torch.cat([avgout, maxout], dim=2)
        return out

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]

    def get_loss(self, x, y):
        predicted = self.forward(x)
        # print("SIZE_X:",predicted)
        # print("SIZE_Y:", y)
        y = y[:, 1:, :]  # (batch, seq, dim_pose)
        # Weighted MSE Loss
        angle_loss = torch.nn.functional.mse_loss(predicted[:, :, :3], y[:, :, :3])
        translation_loss = torch.nn.functional.mse_loss(predicted[:, :, 3:], y[:, :, 3:])
        loss = (100 * angle_loss + translation_loss)
        return loss

    def step(self, x, y, optimizer):
        optimizer.zero_grad()
        loss = self.get_loss(x, y)
        loss.backward()
        if self.clip != None:
            torch.nn.utils.clip_grad_norm(self.rnn.parameters(), self.clip)
        optimizer.step()
        return loss


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(4, 8, 3, 608, 184).to(device)#(4, 8, 3, 484, 304)
    model = MSSTVO('KITTI', par.batch_norm)#VOD
    model = model.to(device)
    y = model(x)
    print(y.shape)
