import torch
import torch.nn as nn
import torch.nn.functional as F
from Res2Net import res2net50_v1b_26w_4s
from TEM import TEM
from PD import PD
from ptflops import get_model_complexity_info


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class PSGNet(nn.Module):

    def __init__(self, channel=32):
        super(PSGNet, self).__init__()

        self.resnet = res2net50_v1b_26w_4s(pretrained=False)

        self.tem2 = TEM(512, channel)
        self.tem3 = TEM(1024, channel)
        self.tem4 = TEM(2048, channel)

        self.agg = PD(channel)

        self.alfa = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.beta = nn.Parameter(torch.FloatTensor(1), requires_grad=True)

        self.sg4_conv1 = BasicConv2d(2048, 256, kernel_size=1)
        self.sg4_conv2 = BasicConv2d(256, 256, kernel_size=5, padding=2)
        self.sg4_conv3 = BasicConv2d(256, 256, kernel_size=5, padding=2)
        self.sg4_conv4 = BasicConv2d(256, 256, kernel_size=5, padding=2)
        self.sg4_conv5 = BasicConv2d(256, 1, kernel_size=1)

        self.sg3_conv1 = BasicConv2d(1024, 64, kernel_size=1)
        self.sg3_conv2 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.sg3_conv3 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.sg3_conv4 = BasicConv2d(64, 1, kernel_size=3, padding=1)

        self.sg2_conv1 = BasicConv2d(512, 64, kernel_size=1)
        self.sg2_conv2 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.sg2_conv3 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.sg2_conv4 = BasicConv2d(64, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x1 = self.resnet.layer1(x)
        x2 = self.resnet.layer2(x1)

        x3 = self.resnet.layer3(x2)
        x4 = self.resnet.layer4(x3)
        x2_tem = self.tem2(x2)
        x3_tem = self.tem3(x3)
        x4_tem = self.tem4(x4)

        sg5 = self.agg(x4_tem, x3_tem, x2_tem)
        m5 = F.interpolate(sg5, scale_factor=8, mode='bilinear')

        s4 = F.interpolate(sg5, scale_factor=0.25, mode='bilinear')
        sig4 = torch.sigmoid(s4)
        x = -1 * sig4 + 1
        sig4 = sig4.expand(-1, 2048, -1, -1).mul(x4)
        x = x.expand(-1, 2048, -1, -1).mul(x4)
        sig4 = self.sg4_conv1(sig4)
        x = self.sg4_conv1(x)
        x = self.alfa * x + self.beta * sig4
        x = F.relu(self.sg4_conv2(x))
        x = F.relu(self.sg4_conv3(x))
        x = F.relu(self.sg4_conv4(x))
        sg4 = self.sg4_conv5(x)
        x = sg4 + s4
        m4 = F.interpolate(x, scale_factor=32, mode='bilinear')

        s3 = F.interpolate(x, scale_factor=2, mode='bilinear')
        sig3 = torch.sigmoid(s3)
        x = -1 * sig3 + 1
        sig3 = sig3.expand(-1, 1024, -1, -1).mul(x3)
        x = x.expand(-1, 1024, -1, -1).mul(x3)
        sig3 = self.sg3_conv1(sig3)
        x = self.sg3_conv1(x)
        x = self.alfa * x + self.beta * sig3
        x = F.relu(self.sg3_conv2(x))
        x = F.relu(self.sg3_conv3(x))
        sg3 = self.sg3_conv4(x)
        x = sg3 + s3
        m3 = F.interpolate(x, scale_factor=16, mode='bilinear')

        s2 = F.interpolate(x, scale_factor=2, mode='bilinear')
        sig2 = torch.sigmoid(s2)
        x = -1 * sig2 + 1
        sig2 = sig2.expand(-1, 512, -1, -1).mul(x2)
        x = x.expand(-1, 512, -1, -1).mul(x2)
        sig2 = self.sg2_conv1(sig2)
        x = self.sg2_conv1(x)
        x = self.alfa * x + self.beta * sig2
        x = F.relu(self.sg2_conv2(x))
        x = F.relu(self.sg2_conv3(x))
        sg2 = self.sg2_conv4(x)
        x = sg2 + s2
        m2 = F.interpolate(x, scale_factor=8, mode='bilinear')

        return m5, m4, m3, m2


if __name__ == '__main__':
    ras = PSGNet().cuda()

    flops, params = get_model_complexity_info(ras, (3, 512, 512), as_strings=True, print_per_layer_stat=True)
    print('flops: ', flops, 'params: ', params)
