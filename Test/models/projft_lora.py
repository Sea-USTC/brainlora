"""
Implementation of LoRA (LoRA: Low-Rank Adaptation of Large Language Models: https://arxiv.org/abs/2106.09685)
Codes are modified from (https://github.com/microsoft/LoRA/blob/main/loralib/layers.py)
"""
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import time

class LoRALayer():
    """
    Base lora class
    """
    def __init__(
            self,
            r,
            lora_alpha,
            lora_dropout = 0.0
         ):
        self.r = r
        self.lora_alpha = lora_alpha
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x

    def reset_parameters(self):
        raise NotImplementedError



class LoRAProj(nn.Linear, LoRALayer):
    def __init__(self, r, lora_alpha, in_features, out_features, lora_dropout = 0.0):
        """
        LoRA class for nn.Linear class
        :param r: low rank dimension
        :param lora_alpha: scaling factor
        :param linear_layer: target nn.Linear layer for applying Lora
        """
        LoRALayer.__init__(self, r, lora_alpha, lora_dropout)
        nn.Linear.__init__(self, in_features, out_features)
        # Lora configuration
        self.lora_A = nn.ParameterList([self.weight.new_zeros((r, in_features)) for _ in range(4)])
        self.lora_B = nn.ParameterList([self.weight.new_zeros((out_features, r)) for _ in range(4)])
        self.scaling = self.lora_alpha / self.r
        self.reset_parameters()

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            for idx in range(4):
                nn.init.kaiming_uniform_(self.lora_A[idx], a=math.sqrt(5))
                nn.init.zeros_(self.lora_B[idx])


    def train(self, mode:bool = True):
        nn.Dropout.train(self, mode)

    def eval(self):
        nn.Dropout.eval(self)


    def forward(self, x, idx):
        result = F.linear(x, self.weight, bias=self.bias)
        out = (x @ self.lora_A[idx].T @ self.lora_B[idx].T*self.scaling)
        result += out
        return result
    
class LoRALinear(nn.Linear, LoRALayer):
    def __init__(self, r, lora_alpha, in_features, out_features, lora_dropout=0.0):
        """
        LoRA class for nn.Linear class
        :param r: low rank dimension
        :param lora_alpha: scaling factor
        :param linear_layer: target nn.Linear layer for applying Lora
        """
        LoRALayer.__init__(self, r, lora_alpha, lora_dropout)
        nn.Linear.__init__(self, in_features, out_features)
        # Lora configuration
        self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
        self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
        self.scaling = self.lora_alpha / self.r
        self.reset_parameters()

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)


    def train(self, mode:bool = True):
        nn.Dropout.train(self, mode)

    def eval(self):
        nn.Dropout.eval(self)


    def forward(self, x):
        result = F.linear(x, self.weight, bias=self.bias)
        out = (x @ self.lora_A.T @ self.lora_B.T*self.scaling)
        result += out
        return result
    