import torch
import torchvision.transforms as tf
import numpy as np
from tqdm import tqdm
from tqdm import tqdm_notebook


def data_read(file, filenum, notebook=False):
    voxel = []
    seg = []
    if notebook:
        tqdmm = tqdm_notebook
    else:
        tqdmm = tqdm
    for i in tqdmm(range(filenum), desc='reading'):
        try:
            data = np.load(file.format(i))
        except FileNotFoundError:
            continue
        try:
            voxel = np.append(voxel, np.expand_dims(data['voxel'], axis=0), axis=0)
            seg = np.append(seg, np.expand_dims(data['seg'], axis=0), axis=0)
        except ValueError:
            voxel = np.expand_dims(data['voxel'], axis=0)
            seg = np.expand_dims(data['seg'], axis=0)
    batch_size = voxel.shape[0]
    return voxel, seg, batch_size


def data_to_tensor(data, dtype=torch.float32, device='cpu'):
    data = torch.from_numpy(data)
    data = data.to(dtype=dtype, device=device)
    return data


def data_mask(data, normalize=True, mask=None, transpose=False):
    if normalize:
        std, mean = torch.std_mean(data[mask])
        data = (data - mean) / std
    if mask is not None:
        data = data.mul(mask.to(dtype=torch.float32))
    if transpose:
        data = data.permute(2, 4, 3)
    return data


def data_crop(data, mask, crop_size=32, masked=True, normalize=True):
    if masked:
        data = data.mul(mask.to(dtype=torch.float32))
    x = mask.sum(dim=2).sum(dim=2).squeeze()
    y = mask.sum(dim=1).sum(dim=2).squeeze()
    z = mask.sum(dim=1).sum(dim=1).squeeze()
    seg_center = torch.zeros(3, dtype=torch.int)
    data_cropped = torch.zeros(data.shape[0], crop_size, crop_size, crop_size)
    for i in range(data.shape[0]):
        x_indices = x[i, :].nonzero()
        seg_center[0] = (x_indices.max() + x_indices.min()) // 2
        seg_center[0] = min(seg_center[0], data.shape[2] - crop_size // 2)
        seg_center[0] = max(seg_center[0], crop_size // 2)
        y_indices = y[i, :].nonzero()
        seg_center[1] = (y_indices.max() + y_indices.min()) // 2
        seg_center[1] = min(seg_center[1], data.shape[2] - crop_size // 2)
        seg_center[1] = max(seg_center[1], crop_size // 2)
        z_indices = z[i, :].nonzero()
        seg_center[2] = (z_indices.max() + z_indices.min()) // 2
        seg_center[2] = min(seg_center[2], data.shape[2] - crop_size // 2)
        seg_center[2] = max(seg_center[2], crop_size // 2)
        data_cropped[i] = data[i, (seg_center[0] - crop_size // 2):(seg_center[0] + crop_size // 2),
                                  (seg_center[1] - crop_size // 2):(seg_center[1] + crop_size // 2),
                                  (seg_center[2] - crop_size // 2):(seg_center[2] + crop_size // 2)]
        if normalize:
            std, mean = torch.std_mean(data_cropped, dim=[1, 2, 3], keepdim=True)
            data_cropped = (data_cropped - mean) / std
    return data_cropped


def data_resize(data, mask, size, normalize=True, masked=False):
    batch_size= data.shape[0]
    if masked:
        data = data.mul(mask.to(dtype=torch.float32))
    result = torch.zeros(batch_size, size, size, size)
    x = mask.sum(dim=2).sum(dim=2).squeeze()
    y = mask.sum(dim=1).sum(dim=2).squeeze()
    z = mask.sum(dim=1).sum(dim=1).squeeze()
    transform = tf.Compose([tf.ToPILImage(), tf.Resize([size, size]), tf.ToTensor()])
    for i in range(batch_size):
        xs = x[i, :].nonzero()
        ys = y[i, :].nonzero()
        zs = z[i, :].nonzero()
        cropped = data[i, xs.min():xs.max(), ys.min():ys.max(), zs.min():zs.max()]
        sizex, sizey, sizez = cropped.shape
        tmp = torch.zeros(sizex, size, size)
        for j in range(sizex):
            tmp[j] = transform(cropped[j])
        for j in range(size):
            result[i, :, j, :] = transform(tmp[:, j, :])
        if normalize:
            std, mean = torch.std_mean(result[i])
            result[i] = (result[i] - mean) / std
    return result


def data_rotate(data, dim=0):
    if dim == 0:
        data = data.permute(1, 3, 2)
    elif dim == 1:
        data = data.permute(3, 1, 2)
    else:
        data = data.permute(2, 1, 3)
    return data
