import torchvision.transforms as transforms
import random
import torchvision.transforms.functional as TF
import PIL
from torch.utils.data import Dataset
import torch
import os


def random_crop(image, label, crop_size):
        
    width, height = image.size  # Assumes the input is a PIL Image, if Tensor, use image.shape[2], image.shape[1]
        
    crop_height, crop_width = crop_size
        
    top = random.randint(0, height - crop_height)
    left = random.randint(0, width - crop_width)
        
    cropped_image = TF.crop(image, top, left, crop_height, crop_width)
    cropped_label = TF.crop(label, top, left, crop_height, crop_width)
    
    return cropped_image, cropped_label

def get_NormTensor(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    
    transform_image = transforms.Compose([        
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    transform_label = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    return transform_image, transform_label

def get_CropNormTensor(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], crop_size=(224, 224)):
    
    transform_image = transforms.Compose([
        transforms.RandomCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
        
    return transform_image

class PathFileUnsupDataset(Dataset):
    def __init__(self, path_file_names, image_transform=None, phase='train'):
        self.path_file_names = path_file_names        
        self.image_transform = image_transform        
        self.phase = phase        
        # self.label_to_id = label_to_id        

    def __len__(self):
        return len(self.path_file_names)

    def __getitem__(self, idx):
        fname = self.path_file_names[idx]
        # fname = os.path.join(self.dataroot, 'JPEGImages', only_fname)
        raw_image = PIL.Image.open(fname)
        image = raw_image.convert("RGB")
                
        idx=fname.find('JPEG')
        only_fname = fname[idx+11:-4]
        
        # if self.phase == 'train':
        #     image, label = random_crop(image, label, self.crop_size)                    
        
        if self.image_transform is not None:
            image = self.image_transform(image)
            # print(torch.max(image), torch.min(image), '\n\n')            
                    
            
            # print(torch.max(label), torch.min(label))            
        
        # if self.label_to_id is not None:
        #     label = self.label_to_id[label]
        return {"image": image,  "name":only_fname}


class PathFileDataset(Dataset):
    def __init__(self, path_file_names, crop_size=(224, 224), image_transform=None, label_trainsform=None, phase='train'):
        self.path_file_names = path_file_names
        self.crop_size = crop_size
        self.image_transform = image_transform
        self.label_transform = label_trainsform        
        self.phase = phase
        # self.label_to_id = label_to_id        

    def __len__(self):
        return len(self.path_file_names)

    def __getitem__(self, idx):
        fname = self.path_file_names[idx]
        # fname = os.path.join(self.dataroot, 'JPEGImages', only_fname)
        raw_image = PIL.Image.open(fname)
        image = raw_image.convert("RGB")
        label_name = fname.replace('JPEGImages', 'SegmentationClass')
        label_name = label_name.replace('jpg', 'png')
        label = PIL.Image.open(label_name)
        
        idx=fname.find('JPEG')
        only_fname = fname[idx+11:-4]
        
        if self.phase == 'train':
            image, label = random_crop(image, label, self.crop_size)
        
        if self.image_transform is not None:
            image = self.image_transform(image)
            # print(torch.max(image), torch.min(image), '\n\n')            
            
        # label = extract_label(fname)
        if self.label_transform is not None:
            label = self.label_transform(label) * 255.0
            label = torch.squeeze(label, dim=0)
            
            # print(torch.max(label), torch.min(label))            
        
        # if self.label_to_id is not None:
        #     label = self.label_to_id[label]
        return {"image": image, "label": label, "name":only_fname}



    
def func_txtfile_read(path_file_list):
    fnames = list()
    with open(path_file_list, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.split()[0]
            line = line + '.jpg'
            fnames.append(line)    
    return fnames

def func_txtpathfile_read(path, file_list):
    fnames = list()    
    path_file_list = os.path.join(path, file_list)
    with open(path_file_list, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.split()[0]
            line = line + '.jpg'            
            fnames.append(os.path.join(path, 'JPEGImages', line))    
    return fnames