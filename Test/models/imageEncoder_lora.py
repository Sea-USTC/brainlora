import numpy as np

from einops import rearrange
from models import resnet_lora as resnet
from models import densenet

import torch
from torch import nn
import torch.nn.functional as F


from io import BytesIO

class ResFeatures(nn.Module):
    def __init__(self, layers) :
        super().__init__()
        self.N = len(layers)
        self.layers = nn.ModuleList(layers)
    '''
    def forward(self, x, idx):
        x = self.conv1(x, idx)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x, idx)
        x = self.layer2(x, idx)
        x = self.layer3(x, idx)
        x = self.layer4(x, idx)
        x = self.conv_seg(x)

        return x
    '''
    def forward(self,x, idx):
        x = self.layers[0](x, idx)
        x = self.layers[1](x)
        x = self.layers[2](x)
        x = self.layers[3](x)
        x = self.layers[4](x, idx)
        x = self.layers[5](x, idx)
        x = self.layers[6](x, idx)
        x = self.layers[7](x, idx)
        return x

class ModelRes(nn.Module):
    def __init__(self, config):
        super(ModelRes, self).__init__()
        self.resnet = self._get_res_base_model(config['model_type'],config['model_depth'],config['input_W'],
                                                config['input_H'],config['input_D'],config['resnet_shortcut'],
                                                config['no_cuda'],config['gpu_id'],config['pretrain_path'],config['out_feature'],config['r_resnet'])
        # num_ftrs = int(self.resnet.fc.in_features/2)
        # self.res_features = nn.Sequential(*list(self.resnet.children())[:-3])
        # self.res_l1 = nn.Linear(num_ftrs, num_ftrs)
        # self.res_l2 = nn.Linear(num_ftrs, 768)

        num_ftrs=int(self.resnet.conv_seg[2].in_features)
        self.res_features = ResFeatures(list(self.resnet.children())[:-1])
        print(self.res_features.children())
        print(self.res_features.named_parameters())
        # # num_ftrs=2048
        out_feature=config['out_feature']
        self.res_l1 = nn.Linear(num_ftrs, num_ftrs)
        self.res_l2 = nn.Linear(num_ftrs, out_feature)

    def _get_res_base_model(self,model_type,model_depth,input_W,input_H,input_D,resnet_shortcut,no_cuda,gpu_id,pretrain_path,out_feature,r):
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
                num_seg_classes=1,
                r=r)
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
        net_dict = model.state_dict()
        model = model.cuda()
        if pretrain_path != 'None':
            print('loading pretrained model {}'.format(pretrain_path))
            pretrain = torch.load(pretrain_path)
            pretrain_dict={}
            for mk, v in pretrain['state_dict'].items():
                k = mk[7:]
                new_k = mk[7:]
                if "conv" in k:
                    conv_idx = k.index("conv")+6
                    new_k = f"{k[:conv_idx]}conv.{k[conv_idx:]}"
                if "layer" in k:
                    lyr_idx = k.index("layer")+7
                    new_k = f"{k[:lyr_idx]}layers.{k[lyr_idx:]}"
                if new_k in net_dict.keys():
                    pretrain_dict[new_k]=v
            print(net_dict.keys())
            print(pretrain['state_dict'].keys())
            print(pretrain_dict.keys())
            #########################
            # pretrain_dict={}
            #########################
            net_dict.update(pretrain_dict) 
            model.load_state_dict(net_dict) 
            print("-------- pre-train model load successfully --------")
        return model

    def forward(self, images, idx):
        # len(images) 4
        # out_embeds = []
        # out_pools = []
        # for i in range(len(images)):
        img = images.float()
        # img = img.cuda(0)
        batch_size = img.shape[0] # 32 1 24 224 224
        res_fea = self.res_features(img, idx) #batch_size,feature_size,patch_num,patch_num 32 512 3 28 28
        res_fea = rearrange(res_fea,'b d n1 n2 n3 -> b (n1 n2 n3) d')
        h = rearrange(res_fea,'b n d -> (b n) d')
        #batch_size,num,feature_size
        # h = h.squeeze()
        x = self.res_l1(h)
        x = F.relu(x)
        x = self.res_l2(x)
        out_embed = rearrange(x,'(b n) d -> b n d',b=batch_size)
        out_pool = torch.mean(out_embed,dim=1)
        # out_embeds.append(out_emb)
        # out_pools.append(out_pool)
        return out_embed,out_pool
    
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

class ModelDense(nn.Module):
    def __init__(self, config):
        super(ModelDense, self).__init__()
        self.densenet = self._get_dense_basemodel(config)
        num_ftrs = int(self.densenet.classifier.in_features)
        self.dense_features = self.densenet.features
        self.dense_l1 = nn.Linear(num_ftrs, num_ftrs)
        self.dense_l2 = nn.Linear(num_ftrs, 768)

            # self.res_features = nn.Sequential(*list(densenet.children())[:-1])
            # # num_ftrs=2048
            # out_feature=config['out_feature']
            # self.res_l1 = nn.Linear(num_ftrs, num_ftrs)
            # self.res_l2 = nn.Linear(num_ftrs, out_feature)
            # self.res_linear1=nn.Linear(out_feature*4,out_feature)
            # self.res_linear2=nn.Linear(out_feature,out_feature)

    def _get_dense_basemodel(self,config):
        assert config['model_type'] in [
            'densenet'
        ]

        if config['model_type'] == 'densenet':
            assert config['model_depth'] in [121,169,201,264]
            model=densenet.generate_model(model_depth=config['model_depth'],
                                        num_classes=config['out_feature'],
                                        n_input_channels=config['in_channels'],
                                        conv1_t_size=config['conv1_t_size'],
                                        conv1_t_stride=config['conv1_t_stride'],
                                        no_max_pool=config['no_max_pool'])
        return model

    def forward(self, img):
        batch_size = img.shape[0]
        dense_fea = self.dense_features(img)#N, 1024, 7,7
        dense_fea = rearrange(dense_fea,'b d n1 n2 -> b (n1 n2) d')
        h = rearrange(dense_fea,'b n d -> (b n) d')
        x = self.dense_l1(h)
        x = F.relu(x)
        x = self.dense_l2(x)
        out_emb = rearrange(x,'(b n) d -> b n d',b=batch_size)
        out_pool = torch.mean(out_emb,dim=1)
        return out_emb,out_pool