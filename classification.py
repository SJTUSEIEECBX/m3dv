# for medical 3d voxel classification project of EE369
import torch
import torch.nn as nn
import torch.utils.data as udata
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from MedicalNets import GoogLeNet
from tensorboardX import SummaryWriter

# super parameters
LR = 0.001
batch_size = 7
EPOCH = 200
writer = SummaryWriter()

# data reading
voxel_train = []
seg_train = []
for i in tqdm(range(584), desc='reading'):
    try:
        data = np.load('data/train_val/candidate{}.npz'.format(i))
    except FileNotFoundError:
        continue
    try:
        voxel_train = np.append(voxel_train, np.expand_dims(data['voxel'], axis=0), axis=0)
        seg_train = np.append(seg_train, np.expand_dims(data['seg'], axis=0), axis=0)
    except ValueError:
        voxel_train = np.expand_dims(data['voxel'], axis=0)
        seg_train = np.expand_dims(data['seg'], axis=0)
training_batch_size = voxel_train.shape[0]
voxel_test = []
seg_test = []
for i in tqdm(range(584), desc='reading'):
    try:
        data = np.load('data/test/candidate{}.npz'.format(i))
    except FileNotFoundError:
        continue
    try:
        voxel_test = np.append(voxel_test, np.expand_dims(data['voxel'], axis=0), axis=0)
        seg_test = np.append(seg_test, np.expand_dims(data['seg'], axis=0), axis=0)
    except ValueError:
        voxel_test = np.expand_dims(data['voxel'], axis=0)
        seg_test = np.expand_dims(data['seg'], axis=0)
test_batch_size = voxel_test.shape[0]
train_label = pd.read_csv('data/train_val.csv').values[:, 1].astype(int)
train_label = torch.from_numpy(train_label)
normalize = transforms.Normalize(mean=[.5], std=[.5])
transform = transforms.Compose([transforms.ToTensor(), normalize])
voxel_train = torch.from_numpy(voxel_train).unsqueeze(1).to(dtype=torch.float32)
voxel_test = torch.from_numpy(voxel_test).unsqueeze(1).to(dtype=torch.float32)
voxel_train -= voxel_train.mean()
voxel_test -= voxel_test.mean()
seg_train = torch.from_numpy(seg_train).unsqueeze(1).to(dtype=torch.float32)
seg_test = torch.from_numpy(seg_test).unsqueeze(1).to(dtype=torch.float32)
masked_voxel_train = voxel_train.mul(seg_train)
masked_voxel_test = voxel_test.mul(seg_test)

validation_batch_size = round(0.2 * training_batch_size)
training_batch_size = training_batch_size - validation_batch_size
train_dataset = udata.TensorDataset(masked_voxel_train, train_label)
masked_train, masked_validate = udata.random_split(train_dataset, [training_batch_size, validation_batch_size])
# voxel_train, voxel_validate = udata.random_split(voxel_train, [training_batch_size, validation_batch_size])
masked_voxel_loader = udata.DataLoader(masked_train, batch_size, shuffle=True)
# voxel_loader = udata.DataLoader(voxel_train, batch_size, shuffle=True)


# train step
losses = torch.zeros(int(EPOCH * (training_batch_size / batch_size)))
net = GoogLeNet().cuda()
optimizer = torch.optim.Adam(net.parameters(), lr=LR)
loss_func = nn.MSELoss()
for epoch in range(EPOCH):
    batch_num = int(training_batch_size / batch_size) + 1
    for j, (voxel, label) in enumerate(tqdm(masked_voxel_loader)):
        voxel = voxel.cuda()
        label = label.cuda()
        prediction = net(voxel)
        loss = loss_func(prediction, label.to(dtype=torch.float32))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses[epoch * batch_num + j] = loss.item()
        writer.add_scalar('scalar/loss', loss.item(), epoch * batch_num + j)
    avg_norm = torch.mean(losses[epoch * batch_num:(epoch + 1) * batch_num])
    print(avg_norm.item())

l1, = plt.plot(list(range(len(losses))), losses)
plt.legend(handles=[l1], labels=['train'], loc='best')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.title('Loss on Train Set')
plt.show()

