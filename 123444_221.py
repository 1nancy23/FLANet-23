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

        self.kernel = torch.tensor(
            [[[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]],
             [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]]).unsqueeze(1).float()




    def forward(self, x):
        x22 = torch.nn.functional.conv2d(x, self.kernel.cuda(), stride=1, padding=1, groups=3)
        x22= x22*2
        return x, x22

def get_filelist(path):
    Filelist = []

    for home, dirs, files in os.walk(path):

        for filename in files:


            Filelist.append(os.path.join(home, filename))



    return Filelist


def cls(path):
    q = get_filelist(path)
    for i in q:
        os.remove(i)


Guss = New_Gauss(3)
path11 = 'F:\\DOTA1.5\\test\\images\\images'
path = "D:\\DATAS\\Image_Test"
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
