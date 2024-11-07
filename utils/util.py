"""This module contains simple helper functions """
from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os
import cv2
from termcolor import cprint
import torchvision.transforms as transforms

def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
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


def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """

    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    if aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    try:
        os.makedirs(path, exist_ok=True)  # exist_ok=True avoids raising an error if the directory exists
        print(f"Directory created or already exists: {path}")
    except Exception as e:
        print(f"An error occurred while creating the directory '{path}': {e}")


def GetPalletePIL(choice, cls_img, w, h):
    if choice == 1:
        b = np.where(cls_img > 127, 255, 0).astype(np.uint8)
        g = np.zeros((w, h), dtype=np.uint8)
        r = np.zeros((w, h), dtype=np.uint8)
    elif choice == 2:
        b = np.zeros((w, h), dtype=np.uint8)
        g = np.where(cls_img > 127, 255, 0).astype(np.uint8)
        r = np.zeros((w, h), dtype=np.uint8)
    elif choice == 3:
        b = np.zeros((w, h), dtype=np.uint8)
        g = np.zeros((w, h), dtype=np.uint8)
        r = np.where(cls_img > 127, 255, 0).astype(np.uint8)
    elif choice == 4:
        b = np.where(cls_img > 127, 255, 0).astype(np.uint8)
        g = np.where(cls_img > 127, 255, 0).astype(np.uint8)
        r = np.where(cls_img > 127, 255, 0).astype(np.uint8)
    else:
        b = np.zeros((w, h), dtype=np.uint8)
        g = np.zeros((w, h), dtype=np.uint8)
        r = np.zeros((w, h), dtype=np.uint8)

    # Combine channels and create an Image
    pallete = Image.merge("RGB", (Image.fromarray(r), Image.fromarray(g), Image.fromarray(b)))
    return pallete

def get_DenormTensor(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    
    # Create the inverse transformation
    inv_mean = [-m / s for m, s in zip(mean, std)]
    inv_std = [1 / s for s in std]
    
    # Define the denormalization transform
    denorm_transform = transforms.Compose([
        transforms.Normalize(mean=inv_mean, std=inv_std)
    ])
    
    return denorm_transform


def pseudo_criterion(criterion, pred, y_origin, y_mixed, lam):
    return lam * criterion(pred, y_origin) + (1 - lam) * criterion(pred, y_mixed)



def print_train_multi_log(iteration, print_epochs, info_list):
    if iteration % print_epochs == 0:
        # cprint('Time:{}||Epoch:{}||EpochIter:{}/{}||Iter:{}||Loss:{:.4f}||Batch_Time:{:.4f}'.format(*info_list), 'green')
        cprint('Time:{}||Epoch:{}||EpochIter:{}/{}||Iter:{}||Loss0:{:.4f}||Loss1:{:.4f}||Loss2:{:.4f}||Loss3:{:.4f}||Batch_Time:{:.4f}'.format(*info_list), 'green')

def print_train_log(iteration, print_epochs, info_list):
    if iteration % print_epochs == 0:
        cprint('Time:{}||Epoch:{}||EpochIter:{}/{}||Iter:{}||Loss:{:.4f}||Batch_Time:{:.4f}'.format(*info_list), 'green')
        
def print_GenImage_train_log(iteration, print_epochs, info_list):
    if iteration % print_epochs == 0:
        cprint('Time:{}||EpochIter:{}/{}||D_fake_Loss:{:.4f}|| D_real_Loss:{:.4f}||G_fake_Loss:{:.4f}||G_L1_Loss:{:.4f}||Batch_Time:{:.4f}'.format(*info_list), 'green')


def print_super_train_log(iteration, print_epochs, info_list):
    if iteration % print_epochs == 0:
        cprint('Time:{}||Epoch:{}||EpochIter:{}/{}||Iter:{}||Loss:{:.4f}||Batch_Time:{:.4f}'.format(*info_list), 'green')
        
def print_train_KD_log(iteration, print_epochs, info_list):
    if iteration % print_epochs == 0:
        # cprint('Time:{}||EpochIter:{}/{}||Loss(Left):{:.4f}||Loss(Right):{:.4f}||Loss(CPS):{:.4f}||Loss(Self):{:.4f}||Batch_Time:{:.4f}'.format(*info_list), 'green')
        cprint('Time:{}||ProjName:{:s}||EpochIter:{}/{}||Loss(Student):{:.4f}||Loss(Const):{:.7f}||Batch_Time:{:.2f}'.format(*info_list), 'green')
                