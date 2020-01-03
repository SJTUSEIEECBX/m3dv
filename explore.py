import torch
import torch.nn as nn
import torch.utils.data as udata
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm_notebook
from GoogLeNet import GoogLeNet
from SimpleNet import SimpleNet
from SimpleNet import CropNet
from SimpleNet import DenseSharp
from tensorboardX import SummaryWriter
from DataProcessing import *
from DataAugment import *


voxel_train, seg_train, total_batch_size = data_read('data/train_val/candidate{}.npz', 58, notebook=False)
voxel_test, seg_test, test_batch_size = data_read('data/test/candidate{}.npz', 58, notebook=False)
label = pd.read_csv('data/train_val.csv').values[:, 1].astype(int)
print('Read Complete!')

label = label[:voxel_train.shape[0]]
mix, mix_label = mixup(voxel_train, label)
plt.imshow(mix[10, 10], cmap=plt.cm.gray)
plt.show()
