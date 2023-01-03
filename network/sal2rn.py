import os
import numpy as np
import random

import torch
import torch.utils.data as dataf
import torch.nn as nn
# from scipy import io
from skimage import io

num_class = 6
NC = 48
FM = 32
channelnumnum = 63
channel_num = 63
filters = [8, 16, 32, 64]
cim_filters = [64, 128]
spe_filters = [8, 16, 32, 64, 96, 128]

class sal2rn(nn.Module):
    def __init__(self):
        super(sal2rn, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=NC,
                out_channels=filters[1],
                kernel_size=3,
                stride=1,
                dilation=1,
            ),
            nn.BatchNorm2d(filters[1]),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=filters[1],
                out_channels=filters[2],
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=2,
            ),
            nn.BatchNorm2d(filters[2]),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=filters[2],
                out_channels=filters[3],
                kernel_size=3,
                stride=1,
                padding=4,
                dilation=5,
            ),
            nn.BatchNorm2d(filters[3]),
            nn.ReLU(),
        )
        self.res1 = nn.Sequential(
            nn.Conv2d(
                in_channels=NC,
                out_channels=filters[1],
                kernel_size=3,
                stride=1,
            ),
            nn.BatchNorm2d(filters[1]),
            nn.ReLU(),
        )
        self.res2 = nn.Sequential(
            nn.Conv2d(
                in_channels=filters[1],
                out_channels=filters[2],
                kernel_size=3,
                stride=1,
            ),
            nn.BatchNorm2d(filters[2]),
            nn.ReLU(),
        )
        self.res3 = nn.Sequential(
            nn.Conv2d(
                in_channels=filters[2],
                out_channels=filters[3],
                kernel_size=3,
                stride=1,
            ),
            nn.BatchNorm2d(filters[3]),
            nn.ReLU(),
        )
        self.afm_conv1 = nn.Sequential(
            nn.Conv2d(filters[3], cim_filters[0], 3, 1, 1),
            nn.BatchNorm2d(cim_filters[0]),
            nn.ReLU(),
        )
        self.afm_conv2 = nn.Sequential(
            nn.Conv2d(filters[3], cim_filters[0], 3, 1, 1),
            nn.BatchNorm2d(cim_filters[0]),
            nn.ReLU(),
        )
        self.afm_conv3 = nn.Sequential(
            nn.Conv2d(cim_filters[1], cim_filters[0], 1, 1, 1),
            nn.BatchNorm2d(cim_filters[0]),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=5),
            nn.Sigmoid(),
        )
        self.out1 = nn.Sequential(
            nn.Linear(in_features=1024, out_features=512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
        )

        self.out1_final = nn.Sequential(
            nn.Linear(256, num_class),
        )

        self.spe_conv1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=11),
            nn.BatchNorm2d(63),
            nn.ReLU(),
        )
        self.spe_conv2 = nn.Sequential(
            nn.Conv1d(in_channels=63, out_channels=63, kernel_size=1, stride=1),
            nn.BatchNorm1d(63),
        )
        self.spe_conv3 = nn.Sequential(
            nn.Conv1d(63, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, 63, kernel_size=1),
            nn.BatchNorm1d(63),
            nn.LeakyReLU(0.2),
        )
        self.linear1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channelnumnum, num_class)
        )
        self.out2 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channelnumnum, num_class)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(1, NC, kernel_size=1, stride=1),
            nn.BatchNorm2d(NC),
            nn.ReLU()
        )
        self.lidar_conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=FM, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(FM),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.5),
            nn.Conv2d(FM, FM * 2, 3, 1, 1),
            nn.BatchNorm2d(FM * 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.5),
            nn.Conv2d(FM * 2, FM * 4, 3, 1, 1),
            nn.BatchNorm2d(FM * 4),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.5),
        )
        self.out3 = nn.Sequential(
            nn.Linear(FM*4, num_class)

        )
    def spatial_channel(self, x1):
        x1_conv = self.conv1(x1)
        x1_res = self.res1(x1)
        x1 = torch.add(x1_conv, x1_res)
        # x1 = torch.cat((x1_conv, x1_res), dim=0)
        x1_conv = self.conv2(x1)
        x1_res = self.res2(x1)
        x1 = torch.add(x1_conv, x1_res)
        x1_conv = self.conv3(x1)
        x1_res = self.res3(x1)
        x1 = torch.add(x1_conv, x1_res)
        afm_conv1 = self.afm_conv1(x1_conv)
        afm_conv2 = self.afm_conv2(x1_res)
        afm_temp = torch.cat((afm_conv1, afm_conv2), dim=1)
        alfa = self.afm_conv3(afm_temp)
        afm_out1 = torch.mul(alfa, afm_conv1)
        beta = 1 - alfa
        afm_out2 = torch.mul(beta, afm_conv2)
        afm_out = torch.add(afm_out1, afm_out2)
        ppp = nn.MaxPool2d(kernel_size=2, stride=1)
        afm_out = ppp(afm_out)
        afm_out = afm_out.view(afm_out.size(0), -1)
        out1 = self.out1(afm_out)
        out1_final = self.out1_final(out1)

        return out1_final

    def spectral_channel(self, x2):
        avgpool = nn.AvgPool2d(kernel_size=11)
        input_x2 = avgpool(x2)
        ga = self.spe_conv1(x2)
        ga = ga.view(ga.size(0), channelnumnum, 1)
        ga = self.spe_conv2(ga)
        ga = ga.view(ga.size(0), channelnumnum, 1)
        input_x2 = input_x2.view(input_x2.size(0), channelnumnum, 1)
        spe_conv2 = ga * input_x2
        spe_out = self.spe_conv3(spe_conv2)
        out2_final = self.out2(spe_out)

        return out2_final

    def lidar_channel(self,x3):
        x3 = self.lidar_conv(x3)
        out3 = x3.view(x3.size(0), -1)
        merge_out = out3
        out3_final = self.out3(merge_out)

        return out3_final
    def forward(self, x1, x2, x3): # x1=spatial, x2=spectral, x3=lidar
        out1_final = self.spatial_channel(x1)
        out2_final = self.spectral_channel(x2)
        out3_final = self.lidar_channel(x3)
        return out1_final, out2_final, out3_final
