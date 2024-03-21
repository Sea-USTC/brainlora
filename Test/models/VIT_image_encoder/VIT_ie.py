import torch
import torch.nn as nn
from .Transformer import TransformerModel
from einops.layers.torch import Rearrange
from einops import repeat,rearrange

class VIT(nn.Module):
    def __init__(
        self,
        img_dim_x,
        img_dim_y,
        img_dim_z,
        channel,
        num_classes,
        patch_dim,
        embedding_dim, # 768
        num_heads,
        num_layers,
        hidden_dim,
        dropout_rate=0.0,
        attn_dropout_rate=0.0
    ):
        super(VIT, self).__init__()

        assert embedding_dim % num_heads == 0
        assert img_dim_x % patch_dim == 0 or img_dim_y % patch_dim == 0 or img_dim_z % patch_dim == 0

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.patch_dim = patch_dim
        self.dropout_rate = dropout_rate
        self.attn_dropout_rate = attn_dropout_rate
        self.channel = channel

        self.num_patches = int((img_dim_x // patch_dim) * (img_dim_y // patch_dim) * (img_dim_z // patch_dim))
        
        self.cls_pos_embedding = nn.Parameter(torch.randn(1,self.num_patches+1,embedding_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1,self.num_patches,embedding_dim))
        
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

        # b,1,24,224,224

        self.proj = nn.Linear(channel*patch_dim*patch_dim*patch_dim,embedding_dim)

    def forward(self, x, as_classifier = False):
        # x.shape b,4,224,224,24
        x = x.float()
        x = x.cuda() # b,1,24,224,224
        b = x.shape[0] 
        x = rearrange(x,'b c (h p1) (w p2) (d p3) -> b (h w d) (p1 p2 p3 c)',p1=self.patch_dim,p2=self.patch_dim,p3=self.patch_dim) # b,25088,2048
        x = self.proj(x) #  b,2352,768
        b,n,_ = x.shape

        if as_classifier:
            cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
            x=torch.cat((cls_tokens,x),dim=1)
            x += self.cls_pos_embedding
            
            x = self.pe_dropout(x)
            # apply transformer
            x, _ = self.transformer(x)

            f = self.mlp_head(x)

            return f
        else:
            x += self.pos_embedding # b,2352,768
            x = self.pe_dropout(x) 
            x, _ = self.transformer(x) # b,2352,768

            return x,x.mean(dim=1)

def VIT_ie(config):
    # VIT_ie(img_dim_x,img_dim_y,img_dim_z,channel,embedding_dim,num_classes) 
    # num_channels = 4
    
    model = VIT(
        img_dim_x=config['input_H'],
        img_dim_y=config['input_W'],
        img_dim_z=config['input_D'],
        channel=config['VIT_channel'],
        num_classes=config['num_classes'],
        patch_dim=8,
        embedding_dim=config['d_model'],
        num_heads=12,
        num_layers=4,
        hidden_dim=768,
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
