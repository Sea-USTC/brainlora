import json
from torch.utils.data import DataLoader
import PIL
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from torchvision import transforms
from PIL import Image
import random
# from dataset.randaugment import RandomAugment
# import skimage.transform as transform
from dataset.augment import *
import nibabel as nib

from .BraTS import nib_load

from augmentation.data_trans import *

class MedKLIP_Dataset(Dataset):
    def __init__(self, data_path, label_path, dis_label_path, report_observe, mode = 'train', augmentation=False, only_global=False,mask_modal=""):
        self.ann = json.load(open(data_path,'r'))
        self.fid_list = list(self.ann)
        self.label_npy = np.load(label_path)
        self.dis_label_npy = np.load(dis_label_path)
        self.report = np.load(report_observe,allow_pickle='True').item()
        self.augmentation = augmentation
        self.only_global = only_global
        self.mask_modal = mask_modal

    def normalize(self,image):
        MIN_BOUND, MAX_BOUND = 0,1000
        image = np.clip(image, MIN_BOUND, MAX_BOUND)
        image = 2 * (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND) - 1
        return image
    
    
    def __getitem__(self, index):
        fid = self.fid_list[index]
        labels = self.label_npy[self.ann[fid]["labels_id"],:]
        dis_labels = self.dis_label_npy[self.ann[fid]["labels_id"],:]
        modal_dic=["DWI","T1WI","T2WI","T2FLAIR"]
        image_sum=[]
        entity = []
        report_entity = []
        if self.augmentation:
            images = []
            for modal in modal_dic:
                if self.ann[fid][modal] == '':
                    image = np.zeros((224,224,24))
                else:
                    data = np.array(nib_load(self.ann[fid][modal], self.ann[fid]['component']), dtype='float32', order='C')
                    # image=self.transform(img_data)
                    image = nnUNet_resample(data,[224,224,24],is_seg=False)
                images.append(image)
            images = np.stack(images, -1)
            mask = images.sum(-1) > 0
            for k in range(len(modal_dic)):
                x = images[..., k]
                y = x[mask]
                mn = y.mean()
                std = y.std()
                x[mask] = (x[mask] - mn) / (std + 1e-8)
                images[..., k] = x
            images = transform(images)
            image_sum = [images[...,k].transpose([2,1,0])[np.newaxis,:] for k in range(len(modal_dic))]
        else:
            for modal in modal_dic:
                if self.ann[fid][modal] == '':
                    image = np.zeros((224,224,24))
                    image = image.transpose([2,1,0])
                    image = image[np.newaxis,:]
                else:
                    data=nib.load(self.ann[fid][modal])
                    img_data=data.get_fdata()
                    if img_data.ndim>3:
                        img_data=img_data[:,:,:,self.ann[fid]['component']]
                    image = nnUNet_resample_and_normalize(img_data,[224,224,24],is_seg=False)
                    image = image.transpose([2,1,0])
                    image=image[np.newaxis,:]
                image_sum.append(image)
        if self.mask_modal != "":
            modal_idx = modal_dic.index(self.mask_modal)
            image_sum[modal_idx] = np.zeros(image_sum[modal_idx].shape)
        if not self.only_global:
            for modal in modal_dic:
                if self.report[fid][modal] == ['']:
                    report_new = ['None']
                    entity.append('[SEP]'.join(report_new))
                else:
                    entity.append('[SEP]'.join(self.report[fid][modal]))
                # if len(self.report[fid][modal]) == 1 and (self.report[fid][modal][0] == "isointensity" or self.report[fid][modal][0] == 'unspecified'):
                #     report_entity.append(modal+' '+self.report[fid][modal][0])
                # else:
                #     report_entity.append('[SEP]'.join(self.report[fid][modal]))
        if 'fuse' in self.report[fid]:
            entity.append('[SEP]'.join(self.report[fid]['fuse']))

        # report = '[SEP]'.join(report_entity)

        return {
            "image": image_sum,
            "label": labels,
            "dis_label": dis_labels,
            # 'index': index_list,
            'entity': entity,
            # 'report':report,
            "fid":fid
            }
    
    
    def __len__(self):
        return len(self.fid_list)

def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset,sampler,bs,n_worker,is_train,collate_fn in zip(datasets,samplers,batch_size,num_workers,is_trains,collate_fns):
        if is_train:
            shuffle = (sampler is None)
            drop_last = True
        else:
            shuffle = False
            drop_last = False
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )              
        loaders.append(loader)
    return loaders    

