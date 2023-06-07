import torch
import torch.nn as nn
import math
import cv2
import os
import torchvision.transforms as transforms
import numpy as np


class New_Gauss(nn.Module):
    def __init__(self, in_channel):
        super(New_Gauss, self).__init__()
        self.GaussBlur1 = self.get_gaussian_kernel(kernel_size=9,channels=in_channel, sigma=10)
        self.GaussBlur2 = self.get_gaussian_kernel(kernel_size=9,channels=in_channel, sigma=20)
        self.kernel = torch.tensor([[[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], [[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]],
                                    [[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]]).unsqueeze(1).float().cuda()  # 锐化卷积核

        self.kernel3 = torch.tensor([[[0, -1, 0], [-1, 9, -1], [0, -1, 0]], [[0, -1, 0], [-1, 9, -1], [0, -1, 0]],
                                    [[0, -1, 0], [-1, 9, -1], [0, -1, 0]]]).unsqueeze(1).float().cuda()  # 锐化卷积核

        self.kernel2 = torch.tensor([[[-0.05, 0.25, 0.25],
                                      [0.25, -0.05, 0.25],
                                      [0.25, 0.25, -0.05]],

                                     [[0.25, -0.05, 0.25],
                                      [-0.05, 0.25, 0.25],
                                      [0.25, -0.05, -0.05]],

                                     [[0.25, 0.25, -0.05],
                                      [0.25, -0.05, 0.25],
                                      [-0.05, 0.25, 0.25]]]).unsqueeze(1).float().cuda()  # 饱和度卷积核

    def get_gaussian_kernel(self, kernel_size=13, sigma=10, channels=3):
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
        gaussian_filter.weight.requires_grad = False

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
        x22 = torch.nn.functional.conv2d(x, self.kernel3, stride=1, padding=1, groups=3)  # 锐化卷积
        # x22 = torch.nn.functional.conv2d(x22, self.kernel2, stride=1, padding=1, groups=3)  # 饱和度卷积
        x1 = torch.sub(self.GaussBlur1(x22), self.GaussBlur2(x22))
        x_big = x22 * x1

        x22 = torch.nn.functional.conv2d(x, self.kernel, stride=1, padding=1, groups=3)  # 锐化卷积
        # x22 = torch.nn.functional.conv2d(x22, self.kernel2, stride=1, padding=1, groups=3)  # 饱和度卷积
        x1 = torch.sub(self.GaussBlur1(x22), self.GaussBlur2(x22))
        x_big2 = x22 * x1

        return x, x_big+x_big2


def get_filelist(path):
    Filelist = []

    for home, dirs, files in os.walk(path):

        for filename in files:
            # 文件名列表，包含完整路径

            Filelist.append(os.path.join(home, filename))

            # # 文件名列表，只包含文件名

            # Filelist.append( filename)

    return Filelist


def cls(path):
    q = get_filelist(path)
    for i in q:
        os.remove(i)


Guss = New_Gauss(3)
path11 = 'F:\\DOTA1.5\\test\\images\\images'
path = "D:\\DATAS\\Image_Test"
# path11 = 'D:\\DATAS\\UCMerced_LandUse\\UCMerced_LandUse\\Images\\storagetanks'  # 数据集路径
# path = "D:\\DATAS\\Image_Test\\"  # 放结果路径
cls(path)
Filelist = get_filelist(path11)

for h, file in enumerate(Filelist):
    print(file)
    imgName = file
    device = torch.device("cuda:0")
    img = cv2.imread(imgName)
    img = np.array(img)
    img = torch.from_numpy(img).float()
    img = img.permute(2, 0, 1)
    img = img.to(device)
    Guss.to(device)

    x, x_big = Guss(img)

    x_big = x_big.permute(1, 2, 0)
    x_big = x_big.cpu()
    x_big = x_big.numpy()
    path33 = os.path.join(path, 'big')
    if not os.path.exists(path33):
        os.makedirs(path33)
    cv2.imwrite(f'{path33}/{file[-9:-4]}.jpg', x_big)
    x = x.permute(1, 2, 0)
    x = x.cpu()
    x = x.numpy()
    path33 = os.path.join(path, 'change')
    if not os.path.exists(path33):
        os.makedirs(path33)
    cv2.imwrite(f'{path33}/{file[-9:-4]}.jpg', x)
