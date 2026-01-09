import torch
import torch.nn as nn
import torch.nn.functional as F

import math


class MultiheadAttention_v3(nn.Module):
    def __init__(self,embed_dim,head_num,kdim=None,vdim=None,dropout=0.0):
        super().__init__()
        self.head_num = head_num
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.head_dim = embed_dim // head_num
        self.dropout = dropout
        self.ln = nn.LayerNorm(256)
        self.bn = nn.BatchNorm1d(256)

        assert self.head_dim * head_num == embed_dim, "embed_dim must be divisible by head_num"

        self.q_proj = nn.Linear(embed_dim,embed_dim)
        self.k_proj = nn.Linear(embed_dim,self.kdim)
        self.v_proj = nn.Linear(embed_dim,self.vdim)

        self.out_proj = nn.Linear(embed_dim,embed_dim)

        if dropout > 0.0:
            self.dropout = nn.Dropout(dropout)

    def forward(self,q,k,v,mask=None):
        # 映射为相同维度
        # [seq_len,batch_size,embed_dim]
        q = self.q_proj(self.ln(q))
        k = self.k_proj(self.ln(k))
        v = self.v_proj(self.ln(v))

        # 分头
        # [seq_len,batch_size,embed_dim] -> [head_num,batch_size,seq_len,head_dim]
        # q的seq_len为tgt_len,k,v的seq_len为src_len，可能不同
        seq_len,batch_size,embed_dim = q.size()
        q = q.view(seq_len,batch_size,self.head_num,self.head_dim).transpose(0,2)
        k = k.view(seq_len,batch_size,self.head_num,self.head_dim).transpose(0,2)
        v = v.view(seq_len,batch_size,self.head_num,self.head_dim).transpose(0,2)

        # attention_weight计算
        # (..,tgt_seq,head_dim)*(..,head_dim,src_len) -> (..,tgt_seq,src_len)
        # (..,tgt_seq,src_len)*(..,src_len,head_dim)->(head_num,batch_size,tgt_len,head_dim)
        attention_weight = torch.matmul(
            q,k.transpose(-2,-1)
        )
        attention_weight = attention_weight / math.sqrt(self.embed_dim)
        # print(attention_weight.shape)

        a,b,c,d = attention_weight.shape
        attention_weight = attention_weight.view(a,b,-1)
        Value_k, index_k = torch.topk(attention_weight, (c*d)//2,dim=-1)
        # print(index_k,(c*d)//2)
        maskM = torch.zeros(a,b,c*d).cuda()
        maskM.scatter_(-1, index_k, 1.)
        # print(maskM[0,3,192930:192940], maskM.shape)
        attention_weight = torch.where(maskM>0,attention_weight,torch.full_like(attention_weight,float("inf")))
        attention_weight = attention_weight.view(a, b, c, d)


        if mask is not None:
            attention_weight = attention_weight.masked_fill(mask,-1e9)
        attention_weight = nn.functional.softmax(attention_weight,dim=-1)
        if self.dropout > 0.0:
            attention_weight = self.dropout(attention_weight,p=self.dropout)
        attention_weight = torch.matmul(attention_weight,v)

        # 合并多头
        attention_weight = attention_weight.transpose(0,2).contiguous().view(seq_len,batch_size,embed_dim)
        attention_weight = self.out_proj(attention_weight)
        return attention_weight

class ImageMultiheadCrossAttention(nn.Module):
    def __init__(self, in_channels, head_num):
        super(ImageMultiheadCrossAttention, self).__init__()
        self.in_channels = in_channels
        self.head_num = head_num
        # self.embed_dim = embed_dim
        self.query_conv = nn.Conv2d(in_channels, in_channels // head_num, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // head_num, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels // head_num, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))  # 可学习缩放因子
        # self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=True)
        # assert self.head_dim * head_num == embed_dim, "embed_dim must be divisible by head_num"
        self.out_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.batchNorm = nn.BatchNorm2d(in_channels)

    def forward(self, q, k, v):
        # 输入 x: (B, C, H, W)
        B, C, H, W = q.size()
        # x = q
        # Hk = H//4
        # Wk = W//4

        for i in range(self.head_num):
            # 生成 Q, K, V
            proj_query = self.query_conv(q).view(B, -1, H * W).permute(0, 2, 1)  # (B, N, C)
            proj_key = self.key_conv(k).view(B, -1, H * W)  # (B, C, N)
            proj_value = self.value_conv(v).view(B, -1, H * W)  # (B, C, N)

            # 注意力矩阵：Q * K^T
            energy = torch.bmm(proj_query, proj_key)  # (B, N, N)
            # energy = torch.div(torch.bmm(proj_query, proj_key),
            #                    torch.sqrt(torch.ones(1).cuda() * self.in_channels))  # (B, N, N)
            attention = F.softmax(energy, dim=-1)  # (B, N, N)

            out = torch.empty(B, self.head_num, C // self.head_num, H * W).cuda()
            # 加权求和 V
            out[:, i, :, :] = torch.bmm(attention, proj_value.permute(0, 2, 1)).permute(0, 2, 1)  # (B, C, N)
            # print(out.shape)
        out = out.view(B, -1, H * W)
        out = out.view(B, C, H, W)
        # print(out.device)
        # out = self.batchNorm(out)

        # # 残差连接 + 缩放因子
        # out = self.gamma * out + x
        # out = 0 * out + x
        return out


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # x = torch.randn(4, 256, 39, 12).to(device)
    # model = ImageMultiheadCrossAttention(256, 8)
    # model = model.to(device)
    # y = model(x, x, x)
    # print(y.shape)

    x = torch.randn(468,4,256).to(device)
    model = MultiheadAttention_v3(256, 8)
    model = model.to(device)
    y = model(x, x, x)
    print(y.shape)