import os
import torch
from torch.utils.data import Dataset
import random
import numpy as np
from torchvision.transforms import transforms
import pickle
from scipy import ndimage
import json
import nibabel as nib
from dataset.augment import *

def pkload(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)


class MaxMinNormalization(object):
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        Max = np.max(image)
        Min = np.min(image)
        image = (image - Min) / (Max - Min)

        return {'image': image, 'label': label}


class Random_Flip(object):
    def __call__(self, sample):
        image = sample['image']
        # label = sample['label']
        if random.random() < 0.5:
            image = np.flip(image, 0)
            # label = np.flip(label, 0)
        if random.random() < 0.5:
            image = np.flip(image, 1)
            # label = np.flip(label, 1)
        if random.random() < 0.5:
            image = np.flip(image, 2)
            # label = np.flip(label, 2)
        sample['image'] = image
        return sample


class GausianNoise(object):
    def __call__(self, sample):
        image = sample['image']
        h,w,d,c = image.shape
        N = np.random.normal(loc=0, scale=0.1,size=(h,w,d,1))
        N = np.repeat(N, c, axis=3)

        image = image + N

        sample['image'] = image
        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        image = sample['image']
        image = np.ascontiguousarray(image.transpose(3, 0, 1, 2))
        # label = sample['label']
        # label = np.ascontiguousarray(label)

        image = torch.from_numpy(image).float()
        # label = torch.from_numpy(label).long()
        sample['image'] = image
        return sample


def transform(sample):
    trans = transforms.Compose([
        # Pad(),
        # Random_rotate(),  # time-consuming
        # Random_Crop(),
        Random_Flip(),
        GausianNoise(),
        ToTensor()
    ])

    return trans(sample)


def transform_valid(sample):
    trans = transforms.Compose([
        # MaxMinNormalization(),
        ToTensor()
    ])

    return trans(sample)

def nib_load(file_name, component):
    if not os.path.exists(file_name):
        print('Invalid file name, can not find the file!')

    proxy = nib.load(file_name)
    data = proxy.get_fdata()
    if data.ndim>3:
        data=data[:,:,:,component]
    proxy.uncache()
    return data

class BraTS(Dataset):
    def __init__(self, csv_path, np_path, mode = 'train'):
        self.ann = json.load(open(csv_path,'r'))
        self.fid_list = list(self.ann)
        self.rad_graph_results = np.load(np_path)
        self.mode=mode
    
    def __getitem__(self, index):
        fid = self.fid_list[index]
        modal_dic=["DWI","T1WI","T2WI","T2FLAIR"]
        images = []
        for modal in modal_dic:
            data = np.array(nib_load(self.ann[fid][modal], self.ann[fid]['component']), dtype='float32', order='C')
            # image=self.transform(img_data)
            image = nnUNet_resample(data,[224,224,24])
            images.append(image)
        images = np.stack(images, -1)
    
        mask = images.sum(-1) > 0
        for k in range(4):
            x = images[..., k]  #
            y = x[mask]
            mn = y.mean()
            std = y.std()
            x[mask] = (x[mask] - mn) / (std + 1e-8)
            images[..., k] = x
        
        class_label = self.rad_graph_results[self.ann[fid]["labels_id"],:,:] # (51, 75)
        labels = self.triplet_extraction(class_label)

        sample = {'image': images, 'label': labels, "fid":fid}
        if self.mode == "train":
            sample = transform(sample)
        else:
            sample = transform_valid(sample)
        return sample
    
    def triplet_extraction(self, class_label):
        exist_labels = np.zeros(class_label.shape[-1]) -1
        for i in range(class_label.shape[1]):
            ### extract the exist label for each entity and maintain -1 if not mentioned. ###
            if 0 in class_label[:,i]:
                exist_labels[i] = 0
                
            if 1 in class_label[:,i]:
                exist_labels[i] = 1

        return exist_labels
    
    def __len__(self):
        return len(self.fid_list)



