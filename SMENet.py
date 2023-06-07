from torch.autograd import Variable
from layers import *
from data import voc
from sub_modules import *
import math
from AMCN import *
import torchvision.transforms as transforms

class New_Gauss(nn.Module):
    def __init__(self, in_channel):
        super(New_Gauss, self).__init__()
        self.GaussBlur1 = self.get_gaussian_kernel(channels=in_channel, sigma=10)
        self.GaussBlur2 = self.get_gaussian_kernel(channels=in_channel, sigma=20)
        self.kernel = torch.tensor([[[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], [[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]],
                                    [[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]]).unsqueeze(1).float()  # 锐化卷积核

        self.kernel3 = torch.tensor([[[0, -1, 0], [-1, 9, -1], [0, -1, 0]], [[0, -1, 0], [-1, 9, -1], [0, -1, 0]],
                                    [[0, -1, 0], [-1, 9, -1], [0, -1, 0]]]).unsqueeze(1).float()  # 锐化卷积核

        self.kernel2 = torch.tensor([[[0.05, -0.25, -0.25],
                                      [-0.25, 0.05, -0.25],
                                      [-0.25, -0.25, 0.05]],

                                     [[-0.25, 0.05, -0.25],
                                      [0.05, -0.25, -0.25],
                                      [-0.25, 0.05, 0.05]],

                                     [[-0.25, -0.25, 0.05],
                                      [-0.25, 0.05, -0.25],
                                      [0.05, -0.25, -0.25]]]).unsqueeze(1).float()  # 饱和度卷积核

    def get_gaussian_kernel(self, kernel_size=5, sigma=10, channels=3):
        # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
        x_coord = torch.arange(kernel_size)
        x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

        mean = (kernel_size - 1) / 2.
        variance = sigma ** 2.

        # Calculate the 2-dimensional gaussian kernel which is
        # the product of two gaussian distributions for two different
        # variables (in this case called x and y)
        gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                          torch.exp(
                              -torch.sum((xy_grid - mean) ** 2., dim=-1) / \
                              (2 * variance)
                          )
        # Make sure sum of values in gaussian kernel equals 1.
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

        # Reshape to 2d depthwise convolutional weight
        gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
        gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

        gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size,
                                    groups=channels, bias=False, padding=kernel_size // 2)

        gaussian_filter.weight.data = gaussian_kernel
        gaussian_filter.weight.requires_grad = True

        return gaussian_filter

    def adjust_saturation(self, tensor, saturation_factor):
        if saturation_factor == 1:
            return tensor
        c, h, w = tensor.size()
        gray_tensor = transforms.functional.rgb_to_grayscale(tensor)
        gray_tensor = gray_tensor.expand(c, h, w)
        return torch.lerp(gray_tensor, tensor, saturation_factor)

    def binarize(self, x, threshold):
        return (x > threshold).to(x.dtype)



    def forward(self, x):
        x22 = torch.nn.functional.conv2d(x, self.kernel3.cuda(), stride=1, padding=1, groups=3)  # 锐化卷积
        x22 = torch.nn.functional.conv2d(x22, self.kernel2.cuda(), stride=1, padding=1, groups=3)  # 饱和度卷积
        x1 = torch.sub(self.GaussBlur1(x22), self.GaussBlur2(x22))
        x_big = x22 * x1

        x22 = torch.nn.functional.conv2d(x, self.kernel.cuda(), stride=1, padding=1, groups=3)  # 锐化卷积
        x22 = torch.nn.functional.conv2d(x22, self.kernel2.cuda(), stride=1, padding=1, groups=3)  # 饱和度卷积
        x1 = torch.sub(self.GaussBlur1(x22), self.GaussBlur2(x22))
        x_big2 = x22 * x1

        return x22, x_big+x_big2


class SMENet(nn.Module):
    def __init__(self, phase, size, base, extras, head, num_classes):  # phase：train or test size:400 base:vgg
        super(SMENet, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.cfg = voc
        self.priorbox = PriorBox(self.cfg)  # return prior_box[cx,cy,w,h]
        with torch.no_grad():
            self.priors = Variable(self.priorbox.forward())
        self.size = size
        self.blocks_fusion = [4, 2, 1]

        self.vgg = nn.ModuleList(base)
        self.vgg2 = nn.ModuleList(base)
        self.L2Norm = L2Norm(512, 20)
        self.extras = nn.ModuleList(extras)
        self.extras_2 = nn.ModuleList(extras)
        self.loc = nn.ModuleList(head[0])  # Location_para
        self.conf = nn.ModuleList(head[1])  # Confidence_Para

        self.resnet_fusion = ResNet_fusion(self.blocks_fusion)
        self.Erase = nn.ModuleList(Erase())
        self.Erase2 = nn.ModuleList(Erase())


        self.Fusion_detailed_information1 = Fusion_detailed_information1()
        self.Fusion_detailed_information2 = Fusion_detailed_information2()
        self.Fusion_detailed_information3 = Fusion_detailed_information3()
        self.Guss = New_Gauss(3)
        self.AMCN1=AMCN(1024,256,50)
        self.AMCN2=AMCN(1024,256,25)
        self.AMCN3=AMCN(1024,256,13)
        self.AMCN4=AMCN(256,256,7)
        self.AMCN5=AMCN(256,256,5)
        self.AMCN6=AMCN(256,256,3)
        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(num_classes, 0, 200, 0.01, 0.45)

    def forward(self, x):

        sources = list()  # save detected feature maps
        sources2 = list()
        loc = list()
        conf = list()
        x,x_big= self.Guss(x)
        
        for k in range(23):
            x = self.vgg[k](x)
        s = self.L2Norm(x)
        sources.append(s)  # extract Conv4_3 layer feature map

        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)
        sources.append(x)

        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)
        # Eliminate irrelevant information
        erase_sources0 = self.Erase[0](sources[0], sources[2])  # p1'
        erase_sources1 = self.Erase[1](sources[1], sources[3])  # p2'

        # Transmit detailed information
        sources[2] = self.Fusion_detailed_information3(sources[0], sources[1], sources[2])  # 1024
        sources[1] = self.Fusion_detailed_information2(sources[0], erase_sources1)  # 1024
        sources[0] = self.Fusion_detailed_information1(erase_sources0)  # 1024
        # sources[0], sources[1], sources[2] = self.change_channels(sources[0], sources[1], sources[2])
        
        
        # x_big
        for k in range(23):
            x_big = self.vgg2[k](x_big)
        s2 = self.L2Norm(x_big)
        sources2.append(s2)  # extract Conv4_3 layer feature map

        for k in range(23, len(self.vgg)):
            x_big = self.vgg2[k](x_big)
        sources2.append(x_big)

        for k, v in enumerate(self.extras_2):
            x_big = F.relu(v(x_big), inplace=True)
            if k % 2 == 1:
                sources2.append(x_big)
        # Eliminate irrelevant information
        erase_sources02 = self.Erase2[0](sources2[0], sources2[2])  # p1'
        erase_sources12 = self.Erase2[1](sources2[1], sources2[3])  # p2'
        # Transmit detailed information
        sources2[2] = self.Fusion_detailed_information3(sources2[0], sources2[1], sources2[2])  # 1024
        sources2[1] = self.Fusion_detailed_information2(sources2[0], erase_sources12)  # 1024
        sources2[0] = self.Fusion_detailed_information1(erase_sources02)  # 1024
        
#         for i in range(len(sources)):
#             sources[i]=torch.cat((sources[i],sources2[i]), 1)
            
#         sources[0], sources[1], sources[2],sources[3],sources[4],sources[5] = self.change_channels(sources[0], sources[1], sources[2],sources[3],sources[4],sources[5])
        sources[0]=self.AMCN1(sources[0],sources2[0])
        sources[1]=self.AMCN2(sources[1],sources2[1])
        sources[2]=self.AMCN3(sources[2],sources2[2])
        sources[3]=self.AMCN4(sources[3],sources2[3])
        sources[4]=self.AMCN5(sources[4],sources2[4])
        sources[5]=self.AMCN6(sources[5],sources2[5])
        
        
        # for i in range(len(sources)):
        #     sources[i] = self.FBS[i](sources[i])
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
    
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)


        if self.phase == "test":
            output = self.detect.forward(
                loc.view(loc.size(0), -1, 4),  # loc preds
                self.softmax(conf.view(conf.size(0), -1, self.num_classes)),  # conf preds
                self.priors.type(type(x.data))  # default boxes
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors
            )
        return output

    # load weights
    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file, map_location=lambda storage, loc: storage), strict=False)
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')
            
def vgg(cfg, i, batch_norm=False):
    layers = []         # cave backbone structure
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2,dilation=1, return_indices=False, ceil_mode=False)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, dilation=1, return_indices=False, ceil_mode=False)]    # the ceiling method (rounding up) is adopted to ensure that the dimension of the data after pooling remains unchanged
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                #layers += [conv2d, nn.GroupNorm(num_groups=4, num_channels=v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v     # The number of output channels of the upper layer is used as the number of input channels of the next layer
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1,dilation=1, return_indices=False, ceil_mode=False)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]    # Save space and perform overlay operation
    return layers

def add_extras(cfg, i, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k + 1],
                                     kernel_size=(1, 3)[flag], stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v

    return layers


base = {
    '400': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '512': [],
}
extras = {
    '400': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    '512': [],
}
mbox = {
    '400': [6, 6, 6, 6, 4, 4],  # number of boxes per feature map location
    '512': [],
}


def _make_divisible(ch, divisor=8, min_ch=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, in_channel, out_channel, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        hidden_channel = in_channel * expand_ratio
        self.use_shortcut = stride == 1 and in_channel == out_channel

        layers = []
        if expand_ratio != 1:
            # 1x1 pointwise conv
            layers.append(ConvBNReLU(in_channel, hidden_channel, kernel_size=1))
        layers.extend([
            # 3x3 depthwise conv
            ConvBNReLU(hidden_channel, hidden_channel, stride=stride, groups=hidden_channel),
            # 1x1 pointwise conv(linear)
            nn.Conv2d(hidden_channel, out_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channel),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_shortcut:
            return x + self.conv(x)
        else:
            return self.conv(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """
    注意：原论文中，在虚线残差结构的主分支上，第一个1x1卷积层的步距是2，第二个3x3卷积层步距是1。
    但在pytorch官方实现过程中是第一个1x1卷积层的步距是1，第二个3x3卷积层步距是2，
    这么做的好处是能够在top1上提升大概0.5%的准确率。
    可参考Resnet v1.5 https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch
    """
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None,
                 groups=1, width_per_group=64):
        super(Bottleneck, self).__init__()

        width = int(out_channel * (width_per_group / 64.)) * groups

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width,
                               kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel * self.expansion,
                               kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 blocks_num,
                 num_classes=1000,
                 include_top=True,
                 groups=1,
                 width_per_group=64):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64

        self.groups = groups
        self.width_per_group = width_per_group

        self.conv1 = nn.Conv2d(32, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = []
        layers.append(block(self.in_channel,
                            channel,
                            downsample=downsample,
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x


def resnet50(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet50-19c8e357.pth
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def multibox(vgg, extra_layers, cfg, num_classes):
    conf_layer = []
    loc_layer = []
    for i in range(len(cfg)):
        conf_layer.append(
            nn.Conv2d(in_channels=256, out_channels=cfg[i] * num_classes, kernel_size=3, stride=1, padding=1))
        loc_layer.append(nn.Conv2d(in_channels=256, out_channels=cfg[i] * 4, kernel_size=3, padding=1, stride=1))
    return vgg, extra_layers, (loc_layer, conf_layer)


# External call interface function
def build_SMENet(phase, size=400, num_classes=11):
    if phase != "test" and phase != "train":
        print("ERROR: Phase: " + phase + " not recognized")
        return
    if size != 400:
        print("ERROR: You specified size " + repr(size) + ". However, " +
              "currently only  (size=300) is supported!")
        return

    base_, extras_, head_ = multibox(vgg(base[str(size)], 3), add_extras(extras[str(size)], 1024), mbox[str(size)], num_classes)
    return SMENet(phase, size, base_, extras_, head_, num_classes)
