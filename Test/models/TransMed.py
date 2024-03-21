import torch
import torch.nn as nn
from .TransBTS.Transformer import TransformerModel
from einops.layers.torch import Rearrange
from einops import repeat,rearrange
import torchvision
from torchvision import models


class TransMedModel(nn.Module):
    def __init__(
        self,
        img_dim_x,
        img_dim_y,
        img_dim_z,
        num_classes,
        patch_dim,
        embedding_dim,
        num_heads,
        num_layers,
        hidden_dim,
        dropout_rate=0.0,
        attn_dropout_rate=0.0
    ):
        super(TransMedModel, self).__init__()

        assert embedding_dim % num_heads == 0
        assert img_dim_x % patch_dim == 0 or img_dim_y % patch_dim == 0 or img_dim_z % patch_dim == 0

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.patch_dim = patch_dim
        self.dropout_rate = dropout_rate
        self.attn_dropout_rate = attn_dropout_rate

        self.num_patches = int(4*(img_dim_z//3)*(img_dim_x//patch_dim)*(img_dim_y//patch_dim))
        
        self.pos_embedding = nn.Parameter(torch.randn(1,self.num_patches+1,embedding_dim))
        
        self.cls_token = nn.Parameter(torch.randn(1,1,embedding_dim))
        
        self.pe_dropout = nn.Dropout(p=self.dropout_rate)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, num_classes)
            )

        self.transformer = TransformerModel(
            embedding_dim,
            num_layers,
            num_heads,
            hidden_dim,
            self.dropout_rate,
            self.attn_dropout_rate,
        )

        self.to_patch_embedding = Rearrange('b c (h p1) (w p2) (d p3) -> (b c d h w) p3 p1 p2',p1=28,p2=28,p3=3)

        resnet34 = models.resnet34(pretrained=False)
        self.num_ftrs=int(resnet34.fc.in_features) # 2048
        self.cnn = nn.Sequential(*list(resnet34.children())[:-1])

        self.proj = nn.Linear(self.num_ftrs, embedding_dim)


    def forward(self, x):
        # x.shape b,4,224,224,24
        b = x.shape[0]
        x = self.to_patch_embedding(x) # 100352,3,8,8

        x = self.cnn(x).squeeze() # 100352,2048

        x = rearrange(x,'(b p) d -> b p d',b=b,d=self.num_ftrs) # b,25088,2048

        x = self.proj(x) # b, 25088,512

        b,n,_ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x=torch.cat((cls_tokens,x),dim=1)
        
        x += self.pos_embedding[:,:(n+1)]
        
        x = self.pe_dropout(x)

        # apply transformer
        x, _ = self.transformer(x) # 4, p, 512

        f = x[:,0,:]
        f = self.mlp_head(f)
        f = f.reshape(-1,1)
        f = torch.sigmoid(f) # b*c, 1

        return f

def TransMed(img_dim_x, img_dim_y, img_dim_z, num_classes):

    # num_channels = 4
    patch_dim = 8

    model = TransMedModel(
        img_dim_x,
        img_dim_y,
        img_dim_z,
        num_classes,
        patch_dim,
        embedding_dim=512,
        num_heads=8,
        num_layers=12,
        hidden_dim=4096,
        dropout_rate=0.1,
        attn_dropout_rate=0.1
    )

    return model


if __name__ == '__main__':
    with torch.no_grad():
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        cuda0 = torch.device('cuda:0')
        x = torch.rand((1, 4, 128, 128, 128), device=cuda0)
        _, model = TransBTS(dataset='brats', _conv_repr=True, _pe_type="learned")
        model.cuda()
        y = model(x)
        print(y.shape)
