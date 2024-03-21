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

class MedKLIP_Dataset(Dataset):
    def __init__(self, csv_path, np_path,config):
        self.ann = json.load(open(csv_path,'r'))
        self.fid_list = list(self.ann)
        self.rad_graph_results = np.load(np_path)
        self.config = config
        self.z = self.config['input_D']
        self.x = self.config['input_W']
        self.y = self.config['input_H']

    def normalize(self,image):
        MIN_BOUND, MAX_BOUND = 0,1000
        image = np.clip(image, MIN_BOUND, MAX_BOUND)
        image = 2 * (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND) - 1
        return image
    
    def __getitem__(self, index):
        fid = self.fid_list[index]
        if self.config['seperate_classifier']:
            class_label = self.rad_graph_results[self.ann[fid]["labels_id"],:] # (13)
        else:
            class_label = self.rad_graph_results[self.ann[fid]["labels_id"]] # (13)
            class_label = [0,1] if class_label == 1 else [1,0]
            class_label = np.array(class_label)
        # labels = np.zeros(class_label.shape[-1]) -1
        # labels, index_list = self.triplet_extraction(class_label)
        modal_dic=["T1CE","T1WI","T2WI","T2FLAIR"]
        image_sum=[]
        for modal in modal_dic:
            if not (self.config['mask_t1ce'] and modal == 'T1CE'):
                data=nib.load(self.ann[fid][modal])
                img_data=data.get_fdata()
                #img_data=self.normalize(img_data)
                #image=downscale(img_data,[224,224,24])
                if self.x // self.z >=3:
                    image = nnUNet_resample_and_normalize(img_data,[self.x,self.y,self.z])
                else:
                    image = nnUNet_resample_and_normalize(img_data,[self.x,self.y,self.z],do_separate_z=False)
                image = image.transpose([2,1,0])
            else:
                image = np.zeros((self.z,self.y,self.x))
            image=image[np.newaxis,:]
            image_sum.append(image)

        return {
            "image": image_sum,
            "label": class_label,
            "fid":fid
            }
    
    def __len__(self):
        return len(self.fid_list)   

