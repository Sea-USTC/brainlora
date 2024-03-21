"""
Implementation of LoRA (LoRA: Low-Rank Adaptation of Large Language Models: https://arxiv.org/abs/2106.09685)
Codes are modified from (https://github.com/microsoft/LoRA/blob/main/loralib/layers.py)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from typing import Optional, List

class LoRALayer():
    def __init__(
        self, 
        r: int, 
        lora_alpha: int, 
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights


class Embedding(nn.Embedding, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        r: int = 0,
        lora_alpha: int = 1,
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Embedding.__init__(self, num_embeddings, embedding_dim, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=0,
                           merge_weights=merge_weights)
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, num_embeddings)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((embedding_dim, r)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()

    def reset_parameters(self):
        nn.Embedding.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.zeros_(self.lora_A)
            nn.init.normal_(self.lora_B)

    def train(self, mode: bool = True):
        nn.Embedding.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0:
                    self.weight.data -= (self.lora_B @ self.lora_A).transpose(0, 1) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0:
                    self.weight.data += (self.lora_B @ self.lora_A).transpose(0, 1) * self.scaling
                self.merged = True
        
    def forward(self, x: torch.Tensor):
        if self.r > 0 and not self.merged:
            result = nn.Embedding.forward(self, x)
            after_A = F.embedding(
                x, self.lora_A.transpose(0, 1), self.padding_idx, self.max_norm,
                self.norm_type, self.scale_grad_by_freq, self.sparse
            )
            result += (after_A @ self.lora_B.transpose(0, 1)) * self.scaling
            return result
        else:
            return nn.Embedding.forward(self, x)
            

class ConvLoRA(nn.Module, LoRALayer):
    def __init__(self, in_channels, out_channels, kernel_size, r=0, lora_alpha=1, lora_dropout=0., merge_weights=False, **kwargs):
        super(ConvLoRA, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)
        assert isinstance(kernel_size, int)
        # Actual trainable parameters
        if r > 0:
            self.lora_A0 = nn.Parameter(
                self.conv.weight.new_zeros((r , in_channels//self.conv.groups * kernel_size * kernel_size * kernel_size),device='cuda')
            ) 
            self.lora_A1 = nn.Parameter(
                self.conv.weight.new_zeros((r , in_channels//self.conv.groups * kernel_size * kernel_size * kernel_size),device='cuda')
            )
            self.lora_A2 = nn.Parameter(
                self.conv.weight.new_zeros((r , in_channels//self.conv.groups * kernel_size * kernel_size * kernel_size),device='cuda')
            )
            self.lora_A3 = nn.Parameter(
                self.conv.weight.new_zeros((r , in_channels//self.conv.groups * kernel_size * kernel_size * kernel_size),device='cuda')
            )
            self.lora_B0 = nn.Parameter(
              self.conv.weight.new_zeros((out_channels, r),device='cuda')
            )
            self.lora_B1 = nn.Parameter(
              self.conv.weight.new_zeros((out_channels, r),device='cuda')
            )
            self.lora_B2 = nn.Parameter(
              self.conv.weight.new_zeros((out_channels, r),device='cuda')
            )
            self.lora_B3 = nn.Parameter(
              self.conv.weight.new_zeros((out_channels, r),device='cuda')
            )
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            ##################################################
            # self.conv.weight.requires_grad = False
            # conv weights still need training in the mixture of lora project
            ##################################################
        self.reset_parameters()
        self.merged = False

    def reset_parameters(self):
        self.conv.reset_parameters()
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A0, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.lora_A1, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.lora_A2, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.lora_A3, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B0)
            nn.init.zeros_(self.lora_B1)
            nn.init.zeros_(self.lora_B2)
            nn.init.zeros_(self.lora_B3)
            
    def train(self, mode=True):
        super(ConvLoRA, self).train(mode)
        if mode:
            if self.merge_weights and self.merged:
                if self.r > 0:
                    # Make sure that the weights are not merged
                    self.conv.weight.data -= (self.lora_B @ self.lora_A).view(self.conv.weight.shape) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                if self.r > 0:
                    # Merge the weights and mark it
                    self.conv.weight.data += (self.lora_B @ self.lora_A).view(self.conv.weight.shape) * self.scaling
                self.merged = True

    def forward(self, x, idx):
        if self.r > 0 and not self.merged:
            loraBidx = "lora_B"+str(idx)
            loraAidx = "lora_A"+str(idx)
            # print(x.get_device())
            # print(getattr(self, loraAidx))
            # print(self.lora_A0)
            # if x.get_device() != getattr(self, loraAidx).get_device():
            #     exit(0)
            return self.conv._conv_forward(
                x, 
                self.conv.weight + (getattr(self, loraBidx) @ getattr(self, loraAidx)).view(self.conv.weight.shape) * self.scaling,
                self.conv.bias
            )
        return self.conv(x)

