import numpy as np
import scipy.ndimage
import scipy
import torch
import torch.utils.data as udata
import torchvision.models


def random_split(voxel, mask, label, ratio=0.2, train_set=1):
    length = voxel.shape[1]
    indices = np.random.permutation(label.shape[0])
    label = label[indices]
    voxel = voxel[indices]
    mask = mask[indices]
    total_size = label.shape[0]
    validate_size = round(total_size * ratio)
    train_size = round((total_size - validate_size) / train_set)
    train_voxel = np.zeros((train_set, train_size, length, length, length))
    train_mask = np.zeros((train_set, train_size, length, length, length))
    train_label = np.zeros((train_set, train_size))
    for i in range(train_set):
        train_voxel[i] = voxel[train_size * i:train_size * (i + 1)]
        train_label[i] = label[train_size * i:train_size * (i + 1)]
        train_mask[i] = mask[train_size * i:train_size * (i + 1)]
    validate_voxel = voxel[train_size * train_set:]
    validate_label = label[train_size * train_set:]
    validate_mask = mask[train_size * train_set:]
    train_size = train_size * train_set
    validate_size = validate_voxel.shape[0]
    return train_voxel, train_mask, train_label, validate_voxel, validate_mask, validate_label, train_size, validate_size


def train_data_process(voxel, mask, label, batch_size):
    dataset = udata.ConcatDataset([
        data_augment(voxel, mask, label, mix_up=False),
        # data_augment(voxel, mask, label, random_move=True, mix_up=False),
        # data_augment(voxel, mask, label, rotation=True, angle=[1, 1, 1], mix_up=False),
        # data_augment(voxel, mask, label, random_move=True, rotation=True, angle=[2, 2, 2], mix_up=False),
        data_augment(voxel, mask, label, rotation=True, angle=[1, 2, 3], mix_up=False),
        # data_augment(voxel, mask, label, random_move=True, rotation=True, angle=[2, 1, 3], mix_up=False),
        # data_augment(voxel, mask, label, random_move=True, mix_up=False),
    ])
    total_size = voxel.shape[0] * 2
    loader = udata.DataLoader(dataset, batch_size, shuffle=True)
    return loader, total_size


def validate_data_process(voxel, mask, label, batch_size, crop_size=32):
    size = voxel.shape[0]
    # voxel = add_mask(voxel, mask)
    center = get_center(mask)
    voxel = crop(voxel, center, crop_size)
    voxel = standardize(voxel)
    dataset = pack(voxel, label)
    test_loader = udata.DataLoader(dataset, batch_size, shuffle=True)
    return test_loader, size


def test_data_process(voxel, mask, batch_size, crop_size=32):
    size = voxel.shape[0]
    # voxel = add_mask(voxel, mask)
    center = get_center(mask)
    voxel = crop(voxel, center, crop_size)
    voxel = standardize(voxel)
    voxel = torch.from_numpy(voxel).to(dtype=torch.float32)
    voxel = voxel.unsqueeze(1)
    # dataset = udata.TensorDataset(voxel)
    test_loader = udata.DataLoader(voxel, batch_size, shuffle=False)
    return test_loader, size


def data_augment(voxel, mask, label, masked=False, mix_up=True,
                 resize=False, factor_range=[0.8, 1.15],
                 random_move=False, max_move=3,
                 crop_size=32,
                 rotation=False, angle=[0, 0, 0],
                 reflection=False, axis=0,
                 standard=True):
    mask = mask.astype(np.float32)
    if masked:
        voxel = add_mask(voxel, mask)

    if resize:
        voxel, mask = random_resize(voxel, mask, factor_range)
    center = get_center(mask)
    if random_move:
        center = random_move_center(center, max_move, voxel.shape[1])
    voxel = crop(voxel, center, crop_size)
    if rotation:
        voxel = rotate(voxel, angle)
    if reflection:
        voxel = flip(voxel, axis)
    if standard:
        voxel = standardize(voxel)
    if mix_up:
        voxel, label = mixup(voxel, label)
    dataset = pack(voxel, label)
    return dataset


def mixup(voxel, label):
    indices = np.random.permutation(voxel.shape[0])
    alpha = np.random.rand(voxel.shape[0], 1, 1, 1) * 0.2 + 0.4
    voxel_mix = voxel[indices]
    label_mix = label[indices]
    voxel_mix = voxel * alpha + voxel_mix * (1 - alpha)
    alpha = alpha.squeeze()
    label_mix = label * alpha + label_mix * (1 - alpha)
    voxel = np.concatenate((voxel, voxel_mix), axis=0)
    label = np.concatenate((label, label_mix), axis=0)
    return voxel, label


def pack(voxel, label):
    voxel = torch.from_numpy(voxel).to(dtype=torch.float32)
    label = torch.from_numpy(label[0:voxel.shape[0]]).to(dtype=torch.float32)
    voxel = voxel.unsqueeze(1)
    dataset = udata.TensorDataset(voxel, label)
    return dataset


def add_mask(voxel, mask):
    voxel = voxel * (mask.astype(np.float32))
    return voxel


def random_resize(data, mask, factor_range=[0.8, 1.15]):
    batch_size = data.shape[0]
    resize_factor = np.random.rand() * (factor_range[1] - factor_range[0]) + factor_range[0]
    size = round(data.shape[1] * resize_factor)
    resized_data = np.zeros((batch_size, size, size, size))
    resized_mask = np.zeros_like(resized_data)
    for i in range(batch_size):
        resized_data[i] = scipy.ndimage.interpolation.zoom(data[i], resize_factor, order=1)
        resized_mask[i] = scipy.ndimage.interpolation.zoom(mask[i], resize_factor, order=0)
    return resized_data, resized_mask


def get_center(mask):
    x = np.sum(mask, axis=(2, 3))
    y = np.sum(mask, axis=(1, 3))
    z = np.sum(mask, axis=(1, 2))
    batch_size = mask.shape[0]
    center = np.zeros((batch_size, 3), dtype=int)
    for i in range(batch_size):
        x_area = np.where(x[i] > 0)
        center[i, 0] = (np.min(x_area) + np.max(x_area)) // 2
        y_area = np.where(y[i] > 0)
        center[i, 1] = (np.min(y_area) + np.max(y_area)) // 2
        z_area = np.where(z[i] > 0)
        center[i, 2] = (np.min(z_area) + np.max(z_area)) // 2
    return center


def random_move_center(center, maxmove=3, bound=100):
    movement = np.random.randint(low=-maxmove, high=maxmove, size=center.shape)
    moved = center + movement
    moved[moved < 0] = 0
    moved[moved > bound - 1] = bound - 1
    return moved


def crop(data, center, size=32):
    bound = data.shape[1]
    batch_size = data.shape[0]
    cropped = np.zeros((batch_size, size, size, size))
    center[center < size // 2] = size // 2
    center[center > bound - size // 2 - size % 2] = bound - size // 2 - size % 2
    for i in range(batch_size):
        low = center[i] - size // 2
        high = center[i] + size // 2 + size % 2
        cropped[i] = data[i, low[0]:high[0], low[1]:high[1], low[2]:high[2]]
    return cropped


def rotate(data, angle):
    batch_size = data.shape[0]
    rotated = np.zeros_like(data)
    for i in range(batch_size):
        X = np.rot90(data[i], angle[0], axes=(0, 1))  # rotate in X-axis
        Y = np.rot90(X, angle[1], axes=(0, 2))  # rotate in Y'-axis
        rotated[i] = np.rot90(Y, angle[2], axes=(1, 2))  # rotate in Z"-axis
    return rotated


def flip(data, axis):
    batch_size = data.shape[0]
    for i in range(batch_size):
        data[i] = np.flip(data[i], axis)
    return data


def standardize(data):
    std = np.std(data)
    mean = np.mean(data)
    data = (data - mean) / std
    return data
