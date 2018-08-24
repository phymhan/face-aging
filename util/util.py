from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os


# Converts a Tensor into an image array (numpy)
# |imtype|: the desired type of the converted numpy array
def tensor2im(input_image, imtype=np.uint8):
    if isinstance(input_image, torch.Tensor):
        image_tensor = input_image.data
    else:
        return input_image
    # only return the first image in a given batch
    image_numpy = image_tensor[0].cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)


def diagnose_network(net, name='network'):
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_age(fname):
    # returns a float
    strlist = fname.split('_')
    age = float(strlist[0])
    return age


def get_age_label(age, binranges):
    L = None
    for L in range(len(binranges) - 1):
        if (age >= binranges[L]) and (age < binranges[L + 1]):
            break
    return L


def upsample2d(inputTensor, targetSize):
    # 2d upsampling of a 4d tensor
    if inputTensor.size(2) == targetSize:
        return inputTensor
    else:
        # return torch.nn.Upsample(size=(targetSize, targetSize), mode='bilinear', align_corners=True)(inputTensor)
        return torch.nn.functional.interpolate(input=inputTensor, size=(targetSize, targetSize), mode='bilinear', align_corners=True)


def expand2d(inputTensor, targetSize):
    # expand a 4d tensor along axis 2 and 3 to targetSize
    return inputTensor.expand(inputTensor.size(0), inputTensor.size(1), targetSize, targetSize)


def expand2d_as(inputTensor, targetTensor):
    # expand a 4d tensor along axis 0, 2 and 3 to those of targetTensor
    return inputTensor.expand(targetTensor.size(0), inputTensor.size(1), targetTensor.size(2), targetTensor.size(3))


# online mean and std, borrowed from https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
# for a new value newValue, compute the new count, new mean, the new M2.
# mean accumulates the mean of the entire dataset
# M2 aggregates the squared distance from the mean
# count aggregates the number of samples seen so far
def update(existingAggregate, newValue):
    (count, mean, M2) = existingAggregate
    count = count + 1
    delta = newValue - mean
    mean = mean + delta / count
    delta2 = newValue - mean
    M2 = M2 + delta * delta2

    return (count, mean, M2)


# retrieve the mean, variance and sample variance from an aggregate
def finalize(existingAggregate):
    (count, mean, M2) = existingAggregate
    (mean, variance, sampleVariance) = (mean, M2/count, M2/(count - 1))
    if count < 2:
        return float('nan')
    else:
        return (mean, variance, sampleVariance)
