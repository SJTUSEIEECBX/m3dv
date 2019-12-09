# for medical 3d voxel classification project of EE369
import torch
import torch.nn as nn
import torch.utils.data as udata
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from GoogLeNet import GoogLeNet
from SimpleNet import SimpleNet
from tensorboardX import SummaryWriter
from DataProcessing import *

# super parameters
LR = 0.001
batch_size = 7
EPOCH = 200
writer = SummaryWriter()

# data reading
voxel_train, seg_train, train_batch_size = data_read('data/train_val/candidate{}.npz', 584)
voxel_test, seg_test, test_batch_size = data_read('data/test/candidate{}.npz', 584)
train_label = pd.read_csv('data/train_val.csv').values[:, 1].astype(int)

train_label = data_to_tensor(train_label)
voxel_train = data_to_tensor(voxel_train)
voxel_test = data_to_tensor(voxel_test)
seg_train = data_to_tensor(seg_train)
seg_test = data_to_tensor(seg_test)

# masked_voxel_train = data_augment(voxel_train, mask=seg_train)
# masked_voxel_test = data_augment(voxel_test, mask=seg_test)

train_data = udata.TensorDataset(voxel_train, train_label)
masked_train_data = udata.TensorDataset(masked_voxel_train, train_label)

validate_batch_size = round(0.2 * train_batch_size)
train_batch_size = train_batch_size - validate_batch_size

masked_train_data, masked_validate_data = udata.random_split(masked_train_data, [train_batch_size, validate_batch_size])
train_data, validate_data = udata.random_split(train_data, [train_batch_size, validate_batch_size])

masked_voxel_loader = udata.DataLoader(masked_train_data, batch_size, shuffle=True)
voxel_loader = udata.DataLoader(train_data, batch_size, shuffle=True)


# train step
losses = torch.zeros(int(EPOCH * (train_batch_size / batch_size)))
net = SimpleNet().cuda()
optimizer = torch.optim.Adam(net.parameters(), lr=LR)
loss_func = nn.MSELoss()
for epoch in range(EPOCH):
    batch_num = int(train_batch_size / batch_size) + 1
    for j, (voxel, label) in enumerate(tqdm(voxel_loader)):
        voxel = voxel.cuda()
        label = label.cuda()
        prediction = net(voxel)
        loss = loss_func(prediction, label.to(dtype=torch.float32))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses[epoch * batch_num + j] = loss.item()
        writer.add_scalar('scalar/loss', loss.item(), epoch * batch_num + j)
    avg_loss = torch.mean(losses[epoch * batch_num:(epoch + 1) * batch_num])
    print(avg_loss.item())

l1, = plt.plot(list(range(len(losses))), losses)
plt.legend(handles=[l1], labels=['train'], loc='best')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.title('Loss on Train Set')
plt.show()

