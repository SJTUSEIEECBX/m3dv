from DataProcessing import *
from DataAugment import *
import numpy as np
import pandas as pd
from SimpleNet import DenseSharp, ResNet
import torch
from tqdm import tqdm


voxel_test, seg_test, test_batch_size = data_read('data/test/candidate{}.npz', 584, notebook=False)
test_loader, test_batch_size = test_data_process(voxel_test, seg_test, 1)

net1 = DenseSharp().cuda()
net2 = DenseSharp().cuda()
net3 = DenseSharp().cuda()
net4 = DenseSharp().cuda()
net1.eval()
net2.eval()
net3.eval()
net4.eval()
torch.cuda.empty_cache()
net1.load_state_dict(torch.load('71/net1_best.pkl'))
net2.load_state_dict(torch.load('71/net2_best.pkl'))
net3.load_state_dict(torch.load('71/net3_best.pkl'))
net4.load_state_dict(torch.load('71/net4_best.pkl'))
prediction = torch.zeros(test_batch_size)
for j, voxel in enumerate(tqdm(test_loader)):
    voxel = voxel.cuda()
    prediction1 = net1(voxel).detach()
    prediction2 = net2(voxel).detach()
    prediction3 = net3(voxel).detach()
    prediction4 = net4(voxel).detach()
    prediction[j] = torch.stack((prediction1, prediction2, prediction3, prediction4), dim=0).mean(dim=0)

submit = pd.read_csv('data/sampleSubmission.csv')
submit['Predicted'] = prediction.numpy()
submit.to_csv('test_result.csv', index=False)
