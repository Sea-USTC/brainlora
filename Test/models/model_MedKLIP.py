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
from models import resnet
from models import resnet,densenet

'''
args.N
args.d_model
args.res_base_model
args.H 
args.num_queries
args.dropout
args.attribute_set_size
'''


class MedKLIP(nn.Module):

    def __init__(self, config, disease_book, mode='train'):
        super(MedKLIP, self).__init__()

        self.mode = mode
        self.d_model = config['d_model']
        # ''' book embedding'''
        with torch.no_grad():
            bert_model = self._get_bert_basemodel(config['text_encoder'],freeze_layers = None).to(disease_book['input_ids'].device)
            self.disease_book = bert_model(input_ids = disease_book['input_ids'],attention_mask = disease_book['attention_mask'])#(**encoded_inputs)
            self.disease_book = self.disease_book.last_hidden_state[:,0,:]
        # self.disease_embedding_layer1 = nn.Linear(768,256)
        # self.disease_embedding_layer2 = nn.Linear(256,768)
        self.cl_fc = nn.Linear(config['out_feature'],768)
        
        self.disease_name = json.load(open('/home/ps/leijiayu/CODE/MedKLIP/Pretrain_MedKLIP/data_file/dis_order.json','r'))
        
        self.keep_class_dim = [i for i in range(len(self.disease_name)) ]
        ''' visual backbone'''
        # self.resnet_dict = {"resnet18": models.resnet18(pretrained=False),
        #                     "resnet50": models.resnet50(pretrained=False)}
        # resnet = self._get_res_basemodel(config['res_base_model'])
        # num_ftrs = int(resnet.fc.in_features/2)
        # self.res_features = nn.Sequential(*list(resnet.children())[:-3])
        # self.res_l1 = nn.Linear(num_ftrs, num_ftrs)
        # self.res_l2 = nn.Linear(num_ftrs, self.d_model)
        if config['model_type']=='resnet':
            resnet=self._get_resnet_model(config['model_type'],config['model_depth'],config['input_W'],
                                                config['input_H'],config['input_D'],config['resnet_shortcut'],
                                                config['no_cuda'],config['gpu_id'],config['pretrain_path'],config['out_feature'])
            self.res_features = nn.Sequential(*list(resnet.children())[:-1])
            # num_ftrs=2048
            num_ftrs=int(resnet.conv_seg[2].in_features)
            out_feature=config['out_feature']
            self.res_l1 = nn.Linear(num_ftrs, num_ftrs)
            self.res_l2 = nn.Linear(num_ftrs, out_feature)
            self.res_linear1=nn.Linear(out_feature*4,out_feature)
            self.res_linear2=nn.Linear(out_feature,out_feature)
        elif config['model_type'] == 'densenet':
            densenet=self._get_densenet_model(config)
            num_ftrs=int(densenet.classifier.in_features)
            self.res_features = nn.Sequential(*list(densenet.children())[:-1])
            # num_ftrs=2048
            out_feature=config['out_feature']
            self.res_l1 = nn.Linear(num_ftrs, num_ftrs)
            self.res_l2 = nn.Linear(num_ftrs, out_feature)
            self.res_linear1=nn.Linear(out_feature*4,out_feature)
            self.res_linear2=nn.Linear(out_feature,out_feature)


        ###################################
        ''' Query Decoder'''
        ###################################

        self.H = config['H'] 
        decoder_layer = TransformerDecoderLayer(self.d_model, config['H'] , 1024,
                                        0.1, 'relu',normalize_before=True)
        decoder_norm = nn.LayerNorm(self.d_model)
        self.decoder = TransformerDecoder(decoder_layer, config['N'] , decoder_norm,
                                  return_intermediate=False)

        # Learnable Queries
        #self.query_embed = nn.Embedding(config['num_queries'] ,self.d_model)
        self.dropout_feas = nn.Dropout(config['dropout'] )

        # Attribute classifier
        self.classifier = nn.Linear(self.d_model,config['attribute_set_size'])

        # # Class classifier
        # self.cls_classifier = nn.Linear(self.d_model,args.num_classes)

        self.apply(self._init_weights)

    def _get_densenet_model(self,config):
        assert config['model_type'] in [
            'densenet'
        ]

        if config['model_type'] == 'densenet':
            assert config['model_depth'] in [121,169,201,264]
            model=densenet.generate_model(model_depth=config['model_depth'],
                                        num_classes=config['num_classes'],
                                        n_input_channels=config['in_channels'],
                                        conv1_t_size=config['conv1_t_size'],
                                        conv1_t_stride=config['conv1_t_stride'],
                                        no_max_pool=config['no_max_pool'])
        return model


    def _get_resnet_model(self,model_type,model_depth,input_W,input_H,input_D,resnet_shortcut,no_cuda,gpu_id,pretrain_path,out_feature):
        assert model_type in [
            'resnet'
        ]

        if model_type == 'resnet':
            assert model_depth in [10, 18, 34, 50, 101, 152, 200]

        if model_depth == 10:
            model = resnet.resnet10(
                sample_input_W=input_W,
                sample_input_H=input_H,
                sample_input_D=input_D,
                shortcut_type=resnet_shortcut,
                no_cuda=no_cuda,
                num_seg_classes=1)
            fc_input = 256
        elif model_depth == 18:
            model = resnet.resnet18(
                sample_input_W=input_W,
                sample_input_H=input_H,
                sample_input_D=input_D,
                shortcut_type=resnet_shortcut,
                no_cuda=no_cuda,
                num_seg_classes=1)
            fc_input = 512
        elif model_depth == 34:
            model = resnet.resnet34(
                sample_input_W=input_W,
                sample_input_H=input_H,
                sample_input_D=input_D,
                shortcut_type=resnet_shortcut,
                no_cuda=no_cuda,
                num_seg_classes=1)
            fc_input = 512
        elif model_depth == 50:
            model = resnet.resnet50(
                sample_input_W=input_W,
                sample_input_H=input_H,
                sample_input_D=input_D,
                shortcut_type=resnet_shortcut,
                no_cuda=no_cuda,
                num_seg_classes=1)
            fc_input = 2048
        elif model_depth == 101:
            model = resnet.resnet101(
                sample_input_W=input_W,
                sample_input_H=input_H,
                sample_input_D=input_D,
                shortcut_type=resnet_shortcut,
                no_cuda=no_cuda,
                num_seg_classes=1)
            fc_input = 2048
        elif model_depth == 152:
            model = resnet.resnet152(
                sample_input_W=input_W,
                sample_input_H=input_H,
                sample_input_D=input_D,
                shortcut_type=resnet_shortcut,
                no_cuda=no_cuda,
                num_seg_classes=1)
            fc_input = 2048
        elif model_depth == 200:
            model = resnet.resnet200(
                sample_input_W=input_W,
                sample_input_H=input_H,
                sample_input_D=input_D,
                shortcut_type=resnet_shortcut,
                no_cuda=no_cuda,
                num_seg_classes=1)
            fc_input = 2048

        model.conv_seg = nn.Sequential(nn.AdaptiveAvgPool3d((1, 1, 1)), nn.Flatten(),
                                   nn.Linear(in_features=fc_input, out_features=out_feature, bias=True))

        # if not no_cuda:
        #     if len(gpu_id) > 1:
        #         model = model.cuda()
        #         model = nn.DataParallel(model, device_ids=gpu_id)
        #         net_dict = model.state_dict()
        #     else:
        #         import os
        #         os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu_id[0])
        #         model = model.cuda()
        #         model = nn.DataParallel(model, device_ids=None)
        #         net_dict = model.state_dict()
        # else:
        #     net_dict = model.state_dict()
        model = model.cuda()
        
        # model = nn.DataParallel(model, device_ids=[0,1])
        # print(model)
        return model

    def _get_bert_basemodel(self, bert_model_name, freeze_layers):
        try:
            model = AutoModel.from_pretrained(bert_model_name)#, return_dict=True)
            print("text feature extractor:", bert_model_name)
        except:
            raise ("Invalid model name. Check the config file and pass a BERT model from transformers lybrary")

        if freeze_layers is not None:
            for layer_idx in freeze_layers:
                for param in list(model.encoder.layer[layer_idx].parameters()):
                    param.requires_grad = False
        return model
    
    def image_encoder(self, images):
        #patch features
        """
        16 torch.Size([16, 1024, 14, 14])
        torch.Size([16, 196, 1024])
        torch.Size([3136, 1024])
        torch.Size([16, 196, 256])
        """
        # batch_size = xis.shape[0]
        # res_fea = self.res_features(xis) #batch_size,feature_size,patch_num,patch_num
        # res_fea = rearrange(res_fea,'b d n1 n2 -> b (n1 n2) d')
        # h = rearrange(res_fea,'b n d -> (b n) d')
        # #batch_size,num,feature_size
        # # h = h.squeeze()
        # x = self.res_l1(h)
        # x = F.relu(x)
        
        
        # x = self.res_l2(x)
        # out_emb = rearrange(x,'(b n) d -> b n d',b=batch_size)
        # return out_emb
        # feature=[]
        # for i in range(4):
        img=images.float()
        img=img.cuda()
        # x=self.res_features(img)
        # print(x.shape)
        batch_size = img.shape[0]
        res_fea = self.res_features(img) #batch_size,feature_size,patch_num,patch_num
        # print(res_fea.shape)
        res_fea = rearrange(res_fea,'b d n1 n2 n3 -> b (n1 n2 n3) d')
        h = rearrange(res_fea,'b n d -> (b n) d')
        #batch_size,num,feature_size
        # h = h.squeeze()
        x = self.res_l1(h)
        x = F.relu(x)
        x = self.res_l2(x)
        out_emb = rearrange(x,'(b n) d -> b n d',b=batch_size)
            # feature.append(out_emb)
        # feature_c=torch.cat(feature,axis=2)
        # feature_c=self.res_linear1(feature_c)
        # out_feature=self.res_linear2(feature_c)
        return out_emb
    
    def forward(self, images):
        print("begin")
        B = images[0].shape[0]
        device = images[0].device
        ''' Visual Backbone '''
        # x = self.image_encoder(images) #b,patch_num,dim 32,2352,768
        # features = x.transpose(0,1) #patch_num b dim 2352,32,768
        disease_book=self.disease_book.clone()
        disease_book=disease_book.to(device)
        # query_embed = self.disease_embedding_layer1(disease_book)
        # query_embed = self.disease_embedding_layer2(query_embed)
        query_embed = disease_book
        query_embed = query_embed.unsqueeze(1).repeat(1, B, 1) # 12,32,768
        # features,ws = self.decoder(query_embed, features, 
        #     memory_key_padding_mask=None, pos=None, query_pos=None) # features 12,32,768
        features=[]
        ws_list=[]
        for i in range(4):
            feature=self.image_encoder(images[i])
            feature=feature.transpose(0,1)
            feature,ws = self.decoder(query_embed, feature, 
                memory_key_padding_mask=None, pos=None, query_pos=None)
            ws_mean=(ws[-4]+ws[-3]+ws[-2]+ws[-1])/4
            features.append(feature)
            ws_list.append(ws_mean)
        out_feature=torch.cat(features,dim=2)
        out_feature=self.res_linear1(out_feature)
        out_feature=self.res_linear2(out_feature)
        out = self.dropout_feas(out_feature)
        print("ws.shape",len(ws),ws[0].shape)
        # ws is output of each transformer layer len(ws) = 4 ws[0].shape = 32,
        # out = self.dropout_feas(features) # 12,32,768
        x= self.classifier(out).transpose(0,1) #B query Atributes 32,12,2
        return x


    @staticmethod
    def _init_weights(module):
        r"""Initialize weights like BERT - N(0.0, 0.02), bias = 0."""

        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)

        elif isinstance(module, nn.MultiheadAttention):
            module.in_proj_weight.data.normal_(mean=0.0, std=0.02)
            module.out_proj.weight.data.normal_(mean=0.0, std=0.02)

        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()