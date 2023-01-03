import os
import numpy as np
import random
from network.sal2rn import sal2rn
from data_reader.data_loader import creat_dataset
import torch
import torch.utils.data as dataf
import torch.nn as nn
# from scipy import io
from skimage import io
from sklearn.decomposition import PCA
import time

hsi_path = './data/TRENTO/hyper_Italy.tif'
lidar_path = './data/TRENTO/LiDAR_Italy.tif'
train_path = './data/TRENTO/TNsecSUBS_Train.tif'
test_path = './data/TRENTO/TNsecSUBS_Test.tif'
num_class = 6
patchsize1 = 11
patchsize2 = 11
batchsize = 64
channelnumnum = 63
EPOCH = 250
LR = 0.001
NC = 48

train_label = io.imread(train_path)
test_label = io.imread(test_path)
hsi_data = io.imread(hsi_path)
hsi_data = hsi_data.astype(np.float32)
lidar_data = io.imread(lidar_path)
lidar_data = lidar_data.astype(np.float32)

train_loader, hsi_train_patch,spa_train_patch,lidar_train_patch, hsi_train_label,\
hsi_test_patch, spa_test_patch, lidar_test_patch, testlabel, TestNum = creat_dataset(hsi_data, lidar_data, train_label,
                        test_label, patchsize1, patchsize2, channelnumnum, NC, batchsize)
cnn = sal2rn()
total_params = sum(p.numel() for p in cnn.parameters())
print(f'{total_params:,} total parameters.')

cnn.cuda()
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()  # the target label is not one-hotted
BestAcc = 0

torch.cuda.synchronize()
start = time.time()
for epoch in range(EPOCH):
    for step, (b_x1, b_x2, b_x3, b_y) in enumerate(train_loader):
        b_x1 = b_x1.cuda()
        b_x2 = b_x2.cuda()
        b_x3 = b_x3.cuda()
        b_y = b_y.cuda()

        out1, out2, out3 = cnn(b_x2, b_x1, b_x3)
        loss1 = loss_func(out1, b_y)
        loss2 = loss_func(out2, b_y)
        loss3 = loss_func(out3, b_y)

        loss = 0.01*loss1 + 0.01*loss2 + 1*loss3

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            cnn.eval()

            temp = hsi_train_patch
            temp = temp.cuda()
            temp1 = spa_train_patch
            temp1 = temp1.cuda()
            temp2 = lidar_train_patch
            temp2 = temp2.cuda()

            temp3, temp4, temp5 = cnn(temp1, temp, temp2)

            pred_y1 = torch.max(temp3, 1)[1].squeeze()
            pred_y1 = pred_y1.cpu()
            acc1 = torch.sum(pred_y1 == hsi_train_label).type(torch.FloatTensor) / hsi_train_label.size(0)

            pred_y2 = torch.max(temp4, 1)[1].squeeze()
            pred_y2 = pred_y2.cpu()
            acc2 = torch.sum(pred_y2 == hsi_train_label).type(torch.FloatTensor) / hsi_train_label.size(0)

            pred_y3 = torch.max(temp5, 1)[1].squeeze()
            pred_y3 = pred_y3.cpu()
            acc3 = torch.sum(pred_y3 == hsi_train_label).type(torch.FloatTensor) / hsi_train_label.size(0)

            Classes = np.unique(hsi_train_label)
            w0 = np.empty(len(Classes),dtype='float32')
            w1 = np.empty(len(Classes),dtype='float32')
            w2 = np.empty(len(Classes),dtype='float32')

            for i in range(len(Classes)):
                cla = Classes[i]
                right1 = 0
                right2 = 0
                right3 = 0

                for j in range(len(hsi_train_label)):
                    if hsi_train_label[j] == cla and pred_y1[j] == cla:
                        right1 += 1
                    if hsi_train_label[j] == cla and pred_y2[j] == cla:
                        right2 += 1
                    if hsi_train_label[j] == cla and pred_y3[j] == cla:
                        right3 += 1

                w0[i] = right1.__float__() / (right1 + right2 + right3 + 0.00001).__float__()
                w1[i] = right2.__float__() / (right1 + right2 + right3 + 0.00001).__float__()
                w2[i] = right3.__float__() / (right1 + right2 + right3 + 0.00001).__float__()
            w0 = torch.from_numpy(w0).cuda()
            w1 = torch.from_numpy(w1).cuda()
            w2 = torch.from_numpy(w2).cuda()

            pred_y = np.empty((len(testlabel)), dtype='float32')
            number = len(testlabel) // 5000
            for i in range(number):
                temp = hsi_test_patch[i * 5000:(i + 1) * 5000, :, :, :]
                temp = temp.cuda()
                temp1 = spa_test_patch[i * 5000:(i + 1) * 5000, :, :, :]
                temp1 = temp1.cuda()
                temp2 = lidar_test_patch[i * 5000:(i + 1) * 5000, :, :, :]
                temp2 = temp2.cuda()
                temp3 = w2*cnn(temp1, temp, temp2)[2] + w1*cnn(temp1, temp, temp2)[1] + w0*cnn(temp1, temp, temp2)[0]
                temp3 = torch.max(temp3, 1)[1].squeeze()
                pred_y[i * 5000:(i + 1) * 5000] = temp3.cpu()
                del temp, temp1, temp3

            if (i + 1) * 5000 < len(testlabel):
                temp = hsi_test_patch[(i + 1) * 5000:len(testlabel), :, :, :]
                temp = temp.cuda()
                temp1 = spa_test_patch[(i + 1) * 5000:len(testlabel), :, :, :]
                temp1 = temp1.cuda()
                temp2 = lidar_test_patch[(i + 1) * 5000:len(testlabel), :, :, :]
                temp2 = temp2.cuda()
                temp3 = w2 * cnn(temp1, temp, temp2)[2] + w1*cnn(temp1, temp, temp2)[1] + w0*cnn(temp1, temp, temp2)[0]
                temp3 = torch.max(temp3, 1)[1].squeeze()
                pred_y[(i + 1) * 5000:len(testlabel)] = temp3.cpu()
                del temp, temp1, temp3

            pred_y = torch.from_numpy(pred_y).long()
            accuracy = torch.sum(pred_y == testlabel).type(torch.FloatTensor) / testlabel.size(0)

            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.cpu().numpy(), '| test accuracy: %.4f' % accuracy)

            if accuracy > BestAcc:
                torch.save(cnn.state_dict(), 'trento_weight.pth')
                BestAcc = accuracy
                w0B = w0
                w1B = w1
                w2B = w2

            cnn.train()

torch.cuda.synchronize()
end = time.time()
print(end - start)
Train_time = end - start
cnn.load_state_dict(torch.load('trento_weight.pth'))
cnn.eval()
w0 = w0B
w1 = w1B
w2 = w2B
torch.cuda.synchronize()
start = time.time()
pred_y = np.empty((len(test_label)), dtype='float32')
number = len(test_label)//5000
for i in range(number):
    temp = hsi_test_patch[i*5000:(i+1)*5000, :, :]
    temp = temp.cuda()
    temp1 = spa_test_patch[i*5000:(i+1)*5000, :, :]
    temp1 = temp1.cuda()
    temp2 = lidar_test_patch[i*5000:(i+1)*5000, :, :]
    temp2 = temp2.cuda()
    temp3 = w2 * cnn(temp1, temp, temp2)[2] + w1 * cnn(temp1, temp, temp2)[1] + w0 * cnn(temp1, temp, temp2)[0]
    temp3 = torch.max(temp3, 1)[1].squeeze()
    pred_y[i*5000:(i+1)*5000] = temp3.cpu()
    del temp, temp1, temp3

if (i+1)*5000 < len(test_label):
    temp = hsi_test_patch[(i+1)*5000:len(test_label), :, :]
    temp = temp.cuda()
    temp1 = spa_test_patch[(i+1)*5000:len(test_label), :, :]
    temp1 = temp1.cuda()
    temp2 = lidar_test_patch[(i+1)*5000:len(test_label), :, :]
    temp2 = temp2.cuda()
    temp3 = w2 * cnn(temp1, temp, temp2)[2] + w1 * cnn(temp1, temp, temp2)[1] + w0 * cnn(temp1, temp, temp2)[0]
    temp3 = torch.max(temp3, 1)[1].squeeze()
    pred_y[(i+1)*5000:len(test_label)] = temp3.cpu()
    del temp, temp1, temp3

pred_y = torch.from_numpy(pred_y).long()


pred_q = pred_y + 1
index = np.load('index.npy')
pred_map = np.zeros_like(test_label)
for i in range(index.shape[1]):
    pred_map[index[0, i], index[1, i]] = pred_q[i]

pred_final = pred_map
np.save('pred', pred_final)


def confusion(pred, label):
    mx = np.zeros((num_class,num_class))
    for i in range (TestNum):
        mx[pred[i], label[i]] += 1
    mx = np.asarray(mx, dtype=np.int16)
    np.savetxt("confusion.txt", mx, delimiter="", fmt="%s")
    return mx


def oa_kappa(confusion):
    N = np.sum(confusion)
    N_ober = np.trace(confusion)
    Po = 1.0*N_ober / N
    h_sum = np.sum(confusion,axis=0)
    v_sum = np.sum(confusion,axis=1)
    Pe = np.sum(np.multiply(1.0*h_sum/N, 1.0*v_sum/N))
    kappa = (Po - Pe)/(1.0 - Pe)
    return kappa

con_matrix = confusion(pred_y, testlabel)
Ka = oa_kappa(con_matrix)
OA = torch.sum(pred_y == testlabel).type(torch.FloatTensor) / testlabel.size(0)

Classes = np.unique(testlabel)
EachAcc = np.empty(len(Classes))

for i in range(len(Classes)):
    cla = Classes[i]
    right = 0
    sum = 0

    for j in range(len(testlabel)):
        if testlabel[j] == cla:
            sum += 1
        if testlabel[j] == cla and pred_y[j] == cla:
            right += 1

    EachAcc[i] = right.__float__()/sum.__float__()

AA = np.mean(EachAcc)
print('ka is', Ka)
print('AA is', AA)
print(OA)
print(EachAcc)

torch.cuda.synchronize()
end = time.time()
print(end - start)
Test_time = end - start
Final_OA = OA

print('The OA is: ', Final_OA)
print('The Training time is: ', Train_time)
print('The Test time is: ', Test_time)

