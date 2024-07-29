import torch
import torch.nn as nn
import torch.nn.functional as F
from model.attention.SEAttention import SEAttention
import numpy as np
from model.attention.SEAttention import SEAttention


class CMT(nn.Module):
    def __init__(self, dim=9, kernel_size=3, stride=1, padding=1, dilation=1):
        super(CMT, self).__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.key_embed = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, bias=False),
            nn.BatchNorm2d(dim),
            nn.LeakyReLU()
        )
        self.value_embed = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(dim)
        )

        factor = 2
        self.attention_embed = nn.Sequential(
            nn.Conv2d(2 * dim, dim // factor, 1, bias=False),
            nn.BatchNorm2d(dim // factor),
            nn.LeakyReLU(),
            nn.Conv2d(dim // factor, self.kernel_size * self.kernel_size * dim, 1, stride=1)
        )

    def forward(self, x):
        bs, c, h, w = x.shape
        k1 = self.key_embed(x)
        v = self.value_embed(x).view(bs, c, -1)

        y = torch.cat([k1, x], dim=1)
        att = self.attention_embed(y)
        att = att.reshape(bs, c, self.kernel_size * self.kernel_size, h, w)
        att = att.mean(2, keepdim=False).view(bs, c, -1)
        k2 = F.softmax(att, dim=-1) * v
        k2 = k2.view(bs, c, h, w)

        return k1 + k2


class LWTD(nn.Module):
    def __init__(self):
        super(LWTD, self).__init__()

        self.increase_dim = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=9, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True)
        )

        self.f1 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=3, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )

        self.f2 = nn.Sequential(
            nn.Conv2d(in_channels=36, out_channels=3, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True)
        )

        self.cmt = nn.Sequential(
            CMT(),
            nn.ReLU(inplace=True)
        )

        self.ca1 = SEAttention(channel=6, reduction=3)
        self.ca2 = SEAttention(channel=9, reduction=9)
        self.ca3 = SEAttention(channel=36, reduction=9)

    def forward(self, x):

        # tensor preparing
        x0 = self.increase_dim(x)
        tensor1, tensor2, tensor3 = torch.split(x0, (3, 3, 3), 1)

        # Block 1 low-stage
        concat1 = self.ca1(torch.cat((x, tensor1), 1))
        conv1 = self.f1(concat1)
        concat2 = self.ca1(torch.cat((conv1, tensor2), 1))
        conv2 = self.f1(concat2)
        concat3 = self.ca1(torch.cat((conv2, tensor3), 1))
        conv3 = self.f1(concat3)
        concat4 = self.ca2(torch.cat((conv1, conv2, conv3), 1))
        cmt1 = self.cmt(concat4)

        c1_1, c1_2, c1_3 = torch.split(cmt1, (3, 3, 3), 1)

        # Block 2 medium-stage
        concat5 = self.ca1(torch.cat((conv3, c1_1), 1))
        conv4 = self.f1(concat5)
        concat6 = self.ca1(torch.cat((conv4, c1_2), 1))
        conv5 = self.f1(concat6)
        concat7 = self.ca1(torch.cat((conv5, c1_3), 1))
        conv6 = self.f1(concat7)
        concat8 = self.ca2(torch.cat((conv4, conv5, conv6), 1))
        cmt2 = self.cmt(concat8)

        c2_1, c2_2, c2_3 = torch.split(cmt2, (3, 3, 3), 1)

        # Block 3 high-stage
        concat9 = self.ca1(torch.cat((conv6, c2_1), 1))
        conv7 = self.f1(concat9)
        concat10 = self.ca1(torch.cat((conv7, c2_2), 1))
        conv8 = self.f1(concat10)
        concat11 = self.ca1(torch.cat((conv8, c2_3), 1))
        conv9 = self.f1(concat11)
        concat12 = self.ca2(torch.cat((conv7, conv8, conv9), 1))
        cmt3 = self.cmt3(concat12)

        concat13 = self.ca3(torch.cat((conv3, cmt1, conv6, cmt2, conv9, cmt3), 1))
        conv10 = self.f2(concat13)

        clean_image = self.relu((conv10 * x) - conv10 + 1)

        return clean_image


if __name__ == '__main__':
    tensor = torch.randn((1, 3, 480, 640)).cuda()

    net = LWTD().cuda()
    x = net(tensor)
    print(x.shape)
    print(net)
