# modified from https://github.com/tensorflow/models/blob/master/research/slim/nets/s3dg.py
#from sklearn.metrics import log_loss
import json
import torch.nn as nn
import torch
import math
import numpy as np  
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from .transformer import *
import torchvision.models as models
from einops import rearrange
from transformers import AutoModel
from models.imageEncoder import ModelRes, ModelDense
from models.VIT_image_encoder.VIT_ie import VIT_ie

from models.tokenization_bert import BertTokenizer


'''
args.N
args.d_model
args.res_base_model
args.H 
args.num_queries
args.dropout
args.attribute_set_size
'''

def get_text_features(model,text_list,tokenizer,device,max_length):
    # text_token =  tokenizer(list(text_list),add_special_tokens=True,max_length=max_length,pad_to_max_length=True,return_tensors='pt').to(device=device)
    target_tokenizer = tokenizer(list(text_list), padding='max_length', truncation=True, max_length=max_length,return_tensors="pt").to(device)
    # text_features = model.encode_text(text_token)
    text_features = model(input_ids = target_tokenizer['input_ids'],attention_mask = target_tokenizer['attention_mask'])#(**encoded_inputs)
    text_features = text_features.last_hidden_state[:,0,:]
    # text_features = F.normalize(text_features, dim=-1)
    return text_features

def _get_bert_basemodel(bert_model_name):
    try:
        model = AutoModel.from_pretrained(bert_model_name)#, return_dict=True)
        print("text feature extractor:", bert_model_name)
    except:
        raise ("Invalid model name. Check the config file and pass a BERT model from transformers lybrary")

    for param in model.parameters():
        param.requires_grad = True

    return model

class chexzero(nn.Module):

    def __init__(self, config):
        super(chexzero, self).__init__()

        # self.mode = mode
        # self.d_model = config['d_model']
        
        # self.cl_fc = nn.Linear(config['out_feature'],768)
        # self.excluded_disease = ['normal']
        # self.disease_name = json.load(open(config['disease_order'],'r'))
        # self.cl_class_dim = [self.disease_name.index(i) for i in self.disease_name if i not in self.excluded_disease ]  
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.config=config
        self.d_model=config['d_model']
        self.res_linear1=nn.Linear(self.d_model*4,self.d_model)
        self.res_linear2=nn.Linear(self.d_model,self.d_model)

        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.config['model_type']== 'resnet':
            self.image_enc =  ModelRes(self.config).to(self.device)
        elif self.config['model_type'] == 'densenet':
            self.image_enc = ModelDense(self.config).to(self.device)
        elif self.config['model_type'] == 'vit':
            self.image_enc = VIT_ie(self.config).to(self.device)
        self.tokenizer = BertTokenizer.from_pretrained(self.config['text_encoder'])
        self.text_enc = _get_bert_basemodel(self.config['text_encoder']).to(self.device)

        # self.initialize_parameters() 
        
        ''' visual backbone'''
    def image_encoder(self,image):
        image_feature_list=[]
        for i in range(4):
            _,out=self.image_enc(image[i])
            image_feature_list.append(out)
        image_feature=torch.cat(image_feature_list,dim=1)
        image_feature=self.res_linear1(image_feature)
        image_feature=self.res_linear2(image_feature)
        return image_feature
    
    def text_encoder(self,text):
        text_feature = get_text_features(self.text_enc,text,self.tokenizer,self.device,max_length=128)
        return text_feature
    
    def forward(self, image, text, ie):
        text = ie.inverse_transform(text.detach().cpu())

        text_features = self.text_encoder(text) # b,768

        if self.config['VIT_channel'] == 1:
            image_features = self.image_encoder(image) # b,768
        elif self.config['VIT_channel'] == 4:
            image_features,image_features_pool = self.image_enc(image)
    
        # return text_features,image_features_pool
        
        # normalized features
        image_features_pool = image_features_pool / image_features_pool.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logit

        return image_features_pool, text_features

    # @staticmethod
    # def _init_weights(module):
    #     r"""Initialize weights like BERT - N(0.0, 0.02), bias = 0."""

    #     if isinstance(module, nn.Linear):
    #         module.weight.data.normal_(mean=0.0, std=0.02)

    #     elif isinstance(module, nn.MultiheadAttention):
    #         module.in_proj_weight.data.normal_(mean=0.0, std=0.02)
    #         module.out_proj.weight.data.normal_(mean=0.0, std=0.02)

    #     elif isinstance(module, nn.Embedding):
    #         module.weight.data.normal_(mean=0.0, std=0.02)
    #         if module.padding_idx is not None:
    #             module.weight.data[module.padding_idx].zero_()