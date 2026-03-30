"""Classifier model used by the pECGMonitor application workflow."""

import torch 
import torch.nn as nn
import torch.nn.functional as F
from ecgtwin.models.clip_model import Net1D 

class ResNetECG(nn.Module):
    """Thin wrapper around the 1D ResNet backbone used for ECG classification."""
    def __init__(self,
                 num_classes,                 
                 ecg_channels=12, 
                 ):
        super().__init__()

        filter_list = [64,128,256,512]
        self.resnet = Net1D(
                in_channels=ecg_channels, 
                base_filters=64, 
                ratio=1, 
                filter_list=filter_list, 
                m_blocks_list=[2,2,2,3], 
                kernel_size=16, 
                stride=2, 
                groups_width=16,
                verbose=False, 
                use_bn=True,
        )

        self.head = nn.Linear(filter_list[-1], num_classes)

    def forward(self, x):
        # input x: (B, C, L) -> (B, L, C) 
        x = torch.transpose(x, 1, 2)
        features = self.resnet(x) 
        logit = F.softmax(self.head(features), dim=1) 

        return logit 
