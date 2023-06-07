import torch
import torch.nn as nn
from torchvision.ops import DeformConv2d
import torch.nn.functional as F


class CroA(nn.Module):
    def __init__(self, in_channel):
        super(CroA, self).__init__()
        self.Seq1_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=256, stride=1, kernel_size=3, padding=1),
            nn.ReLU(True),
        )
        self.Seq1_2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )
        self.Seq2_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=256, stride=1, kernel_size=3, padding=1),
            nn.ReLU(True),
        )
        self.Seq2_2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x1, x2):
        temp1 = self.Seq1_1(x1)
        temp2 = self.Seq2_1(x2)
        x1 = self.Seq1_2(temp1) * temp2
        x2 = self.Seq2_2(temp2) * temp1
        x = x1 + x2
        return x


class Dcnv2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stide=1, padding=1, bias=False):
        super(Dcnv2, self).__init__()
        self.getoffset = nn.Conv2d(in_channels=in_channels, out_channels=2 * kernel_size * kernel_size,
                                   kernel_size=kernel_size
                                   , stride=stide, padding=padding, bias=bias)
        self.deformconv = DeformConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                       padding=padding, stride=stide,
                                       bias=bias)

    def forward(self, x):
        offset = self.getoffset(x)
        out = self.deformconv(x, offset)
        return out


def Upsample(x, y):
    _, _, h1, w1 = x.size()
    result = F.interpolate(y, size=(h1, w1), mode='bilinear')
    return result


class GCA(nn.Module):
    def __init__(self, in_channel_x, in_channel_y):
        super(GCA, self).__init__()
        self.croa = CroA(in_channel_y)
        self.conv1 = nn.Conv2d(in_channels=in_channel_x, out_channels=in_channel_y, stride=1, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=in_channel_x, out_channels=in_channel_y, stride=1, kernel_size=1)

    def forward(self, x1, x2):
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        result = self.croa(x1, x2)
        return result

class CAM(nn.Module):
    def __init__(self,channel):
        super(CAM, self).__init__()
        self.channel=channel
        self.fc = nn.Linear(channel, channel) # 自定义全连接层

    def forward(self, x):
        out = torch.nn.functional.adaptive_avg_pool2d(x, (1,1)) # 进行全局平均池化
        out = torch.flatten(out, 1) # 将特征图展平为特征向量
        out = self.fc(out) # 使用自定义全连接层对特征向量进行线性变换
        self.fc_weights = self.fc.weight # 获取全连接层的权重矩阵
        feature_map = x
        # 使用矩阵乘法代替双for循环
        result = torch.matmul(self.fc_weights.unsqueeze(0), feature_map.view(feature_map.shape[0], feature_map.shape[1], -1))
        result = result.view(feature_map.shape[0], self.fc_weights.shape[0], feature_map.shape[2], feature_map.shape[3])
        # 对每个类别进行归一化处理
        result_max = torch.max(result.view(result.shape[0], result.shape[1], -1), dim=-1)[0].unsqueeze(-1).unsqueeze(-1)
        result_min = torch.min(result.view(result.shape[0], result.shape[1], -1), dim=-1)[0].unsqueeze(-1).unsqueeze(-1)
        result = (result - result_min) / (result_max - result_min + 1e-5)
        return result


class RAB(nn.Module):
    def __init__(self, in_channel):
        super(RAB, self).__init__()
        self.inchannel = in_channel
        self.sa = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=in_channel, stride=1, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=in_channel, out_channels=in_channel, stride=1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=in_channel, stride=1, kernel_size=1),
            nn.Conv2d(in_channels=in_channel, out_channels=in_channel, stride=1, kernel_size=1),
            nn.Conv2d(in_channels=in_channel, out_channels=in_channel, stride=1, kernel_size=1),
            nn.Conv2d(in_channels=in_channel, out_channels=in_channel, stride=1, kernel_size=1),
            nn.Conv2d(in_channels=in_channel, out_channels=in_channel, stride=1, kernel_size=1),
            nn.Conv2d(in_channels=in_channel, out_channels=in_channel, stride=1, kernel_size=1),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=in_channel, stride=1, kernel_size=1),
            nn.Conv2d(in_channels=in_channel, out_channels=in_channel, stride=1, kernel_size=1),
            nn.Conv2d(in_channels=in_channel, out_channels=in_channel, stride=1, kernel_size=1),
            nn.Conv2d(in_channels=in_channel, out_channels=in_channel, stride=1, kernel_size=1),
            nn.Conv2d(in_channels=in_channel, out_channels=in_channel, stride=1, kernel_size=1),
            nn.Conv2d(in_channels=in_channel, out_channels=in_channel, stride=1, kernel_size=1),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=in_channel, stride=1, kernel_size=1),
            nn.Conv2d(in_channels=in_channel, out_channels=in_channel, stride=1, kernel_size=1),
            nn.Conv2d(in_channels=in_channel, out_channels=in_channel, stride=1, kernel_size=1),
            nn.Conv2d(in_channels=in_channel, out_channels=in_channel, stride=1, kernel_size=1),
            nn.Conv2d(in_channels=in_channel, out_channels=in_channel, stride=1, kernel_size=1),
            nn.Conv2d(in_channels=in_channel, out_channels=in_channel, stride=1, kernel_size=1),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=in_channel, stride=1, kernel_size=1),
            nn.Conv2d(in_channels=in_channel, out_channels=in_channel, stride=1, kernel_size=1),
            nn.Conv2d(in_channels=in_channel, out_channels=in_channel, stride=1, kernel_size=1),
            nn.Conv2d(in_channels=in_channel, out_channels=in_channel, stride=1, kernel_size=1),
            nn.Conv2d(in_channels=in_channel, out_channels=in_channel, stride=1, kernel_size=1),
            nn.Conv2d(in_channels=in_channel, out_channels=in_channel, stride=1, kernel_size=1),
        )
        self.CAM = CAM(64)
        self.CAM2 = CAM(64)
        self.CAM3 = CAM(64)
        self.CAM4 = CAM(64)
        self.CAM5 = CAM(64)
        self.relu = nn.ReLU(True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.sa(x)
        x1 = x * x1
        x2 = self.CAM(x)
        x2 = self.relu(x2)
        x2 = self.conv1(x2)
        x2 = self.CAM2(x2)
        x2 = self.relu(x2)
        x2 = self.conv2(x2)
        x2 = self.CAM3(x2)
        x2 = self.relu(x2)
        x2 = self.conv3(x2)
        x2 = self.CAM4(x2)
        x2 = self.relu(x2)
        x2 = self.conv4(x2)
        x2 = self.CAM5(x2)
        x4 = x1 + x2
        x4 = x + x4
        return x4

class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        # mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x

class AMCN(nn.Module):
    def __init__(self, in_channel_x,out_channel,size):
        super(AMCN, self).__init__()
        self.GCA = GCA(in_channel_x,512)
        self.RAB1 = RAB(64)
        self.RAB2 = RAB(64)
        self.RAB3 = RAB(64)
        self.RAB4 = RAB(64)
        self.conv1 = nn.Conv2d(in_channels=256, out_channels=64, stride=1, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels=192, out_channels=64, stride=1, kernel_size=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=out_channel, stride=1, kernel_size=1)
        self.conv5 = nn.Conv2d(in_channels=in_channel_x, out_channels=out_channel, stride=1, kernel_size=1)
        self.RDB1 = ResidualDenseBlock_5C(64, 256)
        self.RDB2 = ResidualDenseBlock_5C(64, 256)
        self.RDB3 = ResidualDenseBlock_5C(64, 256)
        
    def forward(self, x1, x2):
        x2 = self.GCA(x1, x2)
        x2 = self.conv1(x2)
        x2 = self.RDB1(x2)
        x2 = self.RAB1(x2)
        x3 = x2
        x2 = self.RDB2(x2)
        x2 = self.RAB2(x2)
        x4 = x2
        x2 = self.RDB3(x2)
        x2 = self.RAB3(x2)
        x2 = torch.cat((x3,x4,x2), dim=1)
        x2 = self.conv3(x2)
        x2 = self.RAB4(x2)
        x2 = self.conv4(x2)
        x1 = self.conv5(x1)
        x2 = x1+x2
        
        return x2
