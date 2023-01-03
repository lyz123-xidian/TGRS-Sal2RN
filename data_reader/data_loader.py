import os
import numpy as np
import random
import torch
import torch.utils.data as dataf
import torch.nn as nn
# from scipy import io
from skimage import io
from sklearn.decomposition import PCA

def hsi_norm(hsi_data, channelnumnum, patchsize1, NC):
    [m, n, l] = hsi_data.shape
    for i in range(l):
        minimal = hsi_data[:, :, i].min()
        maximal = hsi_data[:, :, i].max()
        hsi_data[:, :, i] = (hsi_data[:, :, i] - minimal) / (maximal - minimal)

    x = np.empty((176, 610, channelnumnum), dtype='float32')
    for i in range(channelnumnum):
        temp = hsi_data[:, :, i]
        pad_width = np.floor(patchsize1 / 2)
        pad_width = np.int(pad_width)
        temp2 = np.pad(temp, pad_width, 'symmetric')
        x[:, :, i] = temp2
    PC = np.reshape(hsi_data, (m * n, l))
    pca = PCA(n_components=NC, copy=True, whiten=False)
    PC = pca.fit_transform(PC)
    PC = np.reshape(PC, (m, n, NC))
    temp = PC[:, :, 0]
    pad_width = np.floor(patchsize1 / 2)
    pad_width = np.int(pad_width)
    temp2 = np.pad(temp, pad_width, 'symmetric')
    [m2, n2] = temp2.shape
    x1 = np.empty((m2, n2, NC), dtype='float32')

    for i in range(NC):
        temp = PC[:, :, i]
        pad_width = np.floor(patchsize1 / 2)
        pad_width = np.int(pad_width)
        temp2 = np.pad(temp, pad_width, 'symmetric')
        x1[:, :, i] = temp2
    return x, x1, pad_width

def lidar_norm(lidar_data, patchsize2):
    minimal = lidar_data.min()
    maximal = lidar_data.max()
    Data2 = (lidar_data - minimal) / (maximal - minimal)
    x2 = Data2
    pad_width2 = np.floor(patchsize2 / 2)
    pad_width2 = np.int(pad_width2)
    temp2 = np.pad(x2, pad_width2, 'symmetric')
    x2 = temp2
    return x2


def create_patch(x,pad_width, train_label, TsLabel, channel, patchsize1):
    [ind1, ind2] = np.where(train_label != 0)
    TrainNum = len(ind1)
    TrainPatch = np.empty((TrainNum, channel, patchsize1, patchsize1), dtype='float32')
    TrainLabel = np.empty(TrainNum)
    ind3 = ind1 + pad_width
    ind4 = ind2 + pad_width
    for i in range(len(ind1)):
        patch = x[(ind3[i] - pad_width):(ind3[i] + pad_width + 1), (ind4[i] - pad_width):(ind4[i] + pad_width + 1), :]
        patch = np.reshape(patch, (patchsize1 * patchsize1, channel))
        patch = np.transpose(patch)
        patch = np.reshape(patch, (channel, patchsize1, patchsize1))
        TrainPatch[i, :, :, :] = patch
        patchlabel = train_label[ind1[i], ind2[i]]
        TrainLabel[i] = patchlabel
    [ind1, ind2] = np.where(TsLabel != 0)
    np.save('index.npy', [ind1, ind2])
    TestNum = len(ind1)
    TestPatch = np.empty((TestNum, channel, patchsize1, patchsize1), dtype='float32')
    TestLabel = np.empty(TestNum)
    ind3 = ind1 + pad_width
    ind4 = ind2 + pad_width
    for i in range(len(ind1)):
        patch = x[(ind3[i] - pad_width):(ind3[i] + pad_width + 1), (ind4[i] - pad_width):(ind4[i] + pad_width + 1), :]
        patch = np.reshape(patch, (patchsize1 * patchsize1, channel))
        patch = np.transpose(patch)
        patch = np.reshape(patch, (channel, patchsize1, patchsize1))
        TestPatch[i, :, :, :] = patch
        testpatchlabel = TsLabel[ind1[i], ind2[i]]
        TestLabel[i] = testpatchlabel

    TrainPatch = torch.from_numpy(TrainPatch)
    TrainLabel = torch.from_numpy(TrainLabel) - 1
    TrainLabel = TrainLabel.long()
    TestPatch = torch.from_numpy(TestPatch)
    TestLabel = torch.from_numpy(TestLabel) - 1
    TestLabel = TestLabel.long()

    return TrainPatch, TrainLabel, TestPatch, TestLabel, TestNum

def create_lidar_patch(x,pad_width, train_label, TsLabel, channel, patchsize1):
    [ind1, ind2] = np.where(train_label != 0)
    TrainNum = len(ind1)
    TrainPatch = np.empty((TrainNum, channel, patchsize1, patchsize1), dtype='float32')
    TrainLabel = np.empty(TrainNum)
    ind3 = ind1 + pad_width
    ind4 = ind2 + pad_width
    for i in range(len(ind1)):
        patch = x[(ind3[i] - pad_width):(ind3[i] + pad_width + 1), (ind4[i] - pad_width):(ind4[i] + pad_width + 1)]
        patch = np.reshape(patch, (patchsize1 * patchsize1, channel))
        patch = np.transpose(patch)
        patch = np.reshape(patch, (channel, patchsize1, patchsize1))
        TrainPatch[i, :, :, :] = patch
        patchlabel = train_label[ind1[i], ind2[i]]
        TrainLabel[i] = patchlabel
    [ind1, ind2] = np.where(TsLabel != 0)
    np.save('index.npy', [ind1, ind2])
    TestNum = len(ind1)
    TestPatch = np.empty((TestNum, channel, patchsize1, patchsize1), dtype='float32')
    TestLabel = np.empty(TestNum)
    ind3 = ind1 + pad_width
    ind4 = ind2 + pad_width
    for i in range(len(ind1)):
        patch = x[(ind3[i] - pad_width):(ind3[i] + pad_width + 1), (ind4[i] - pad_width):(ind4[i] + pad_width + 1)]
        patch = np.reshape(patch, (patchsize1 * patchsize1, channel))
        patch = np.transpose(patch)
        patch = np.reshape(patch, (channel, patchsize1, patchsize1))
        TestPatch[i, :, :, :] = patch
        testpatchlabel = TsLabel[ind1[i], ind2[i]]
        TestLabel[i] = testpatchlabel

    TrainPatch = torch.from_numpy(TrainPatch)
    TrainLabel = torch.from_numpy(TrainLabel) - 1
    TrainLabel = TrainLabel.long()
    TestPatch = torch.from_numpy(TestPatch)
    TestLabel = torch.from_numpy(TestLabel) - 1
    TestLabel = TestLabel.long()

    return TrainPatch, TrainLabel, TestPatch, TestLabel

def creat_dataloader(hsipatch, spapatch, lidarpatch, labelpatch, batchsize):
    dataset = dataf.TensorDataset(hsipatch, spapatch, lidarpatch, labelpatch)
    train_loader = dataf.DataLoader(dataset, batch_size=batchsize, shuffle=True)

    return train_loader

def creat_dataset(hsi_data, lidar_data, train_label, test_label, patchsize1, patchsize2, channelnumnum, NC, batchsize):
    hsi_data_norm, spa_data_norm, pad_width = hsi_norm(hsi_data, channelnumnum, patchsize1, NC)
    lidar_data_norm = lidar_norm(lidar_data, patchsize2)
    hsi_train_patch, hsi_train_label, hsi_test_patch, hsi_test_label, TestNum = create_patch(hsi_data_norm, pad_width, train_label, test_label, channelnumnum, patchsize1)
    spa_train_patch, spa_train_label, spa_test_patch, spa_test_label, TestNum  = create_patch(spa_data_norm, pad_width, train_label, test_label, NC, patchsize1)
    lidar_train_patch, lidar_train_label, lidar_test_patch, lidar_test_label  = create_lidar_patch(lidar_data_norm, pad_width, train_label, test_label, 1, patchsize1)
    train_loader = creat_dataloader(hsi_train_patch, spa_train_patch, lidar_train_patch, hsi_train_label, batchsize)

    return train_loader, hsi_train_patch,spa_train_patch,lidar_train_patch, hsi_train_label,hsi_test_patch, spa_test_patch, lidar_test_patch, hsi_test_label, TestNum





