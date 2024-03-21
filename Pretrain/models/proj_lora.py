"""
Implementation of LoRA (LoRA: Low-Rank Adaptation of Large Language Models: https://arxiv.org/abs/2106.09685)
Codes are modified from (https://github.com/microsoft/LoRA/blob/main/loralib/layers.py)
"""
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import time

class LoRALayer(nn.Module):
    """
    Base lora class
    """
    def __init__(
            self,
            r,
            lora_alpha,
         ):
        super().__init__()
        self.r = r
        self.lora_alpha = lora_alpha

    def reset_parameters(self):
        raise NotImplementedError

    def train(self, mode:bool = True):
        raise NotImplementedError

    def eval(self):
        raise NotImplementedError


class LoRAProj(LoRALayer):
    def __init__(self, r, lora_alpha, in_features, out_features):
        """
        LoRA class for nn.Linear class
        :param r: low rank dimension
        :param lora_alpha: scaling factor
        :param linear_layer: target nn.Linear layer for applying Lora
        """
        super().__init__(r, lora_alpha)
        self.linear = nn.Linear(in_features, out_features)
        # Lora configuration
        self.lora_A = nn.ParameterList([self.linear.weight.new_zeros((r, in_features)) for _ in range(4)])
        self.lora_B = nn.ParameterList([self.linear.weight.new_zeros((out_features, r)) for _ in range(4)])
        self.scaling = self.lora_alpha / self.r
        self.reset_parameters()

    def reset_parameters(self):
        for idx in range(4):
            nn.init.kaiming_uniform_(self.lora_A[idx], a=math.sqrt(5))
            nn.init.zeros_(self.lora_B[idx])


    def train(self, mode:bool = True):
        self.linear.train(mode)


    def eval(self):
        self.linear.eval()


    def forward(self, x, idx):
        result = F.linear(x, self.linear.weight, bias=self.linear.bias)
        out = (x @ self.lora_A[idx].T @ self.lora_B[idx].T)
        result += out
        return result
    

