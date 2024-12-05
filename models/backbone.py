# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
import models.resnet_ as resnet
import models.vgg_ as models

class BackboneBase_VGG(nn.Module):
    def __init__(self, backbone: nn.Module, num_channels: int, name: str, return_interm_layers: bool):
        super().__init__()
        features = list(backbone.features.children())
        if return_interm_layers:
            if name == 'vgg16_bn':
                self.body1 = nn.Sequential(*features[:13])
                self.body2 = nn.Sequential(*features[13:23])
                self.body3 = nn.Sequential(*features[23:33])
                self.body4 = nn.Sequential(*features[33:43])
            else:
                self.body1 = nn.Sequential(*features[:9])
                self.body2 = nn.Sequential(*features[9:16])
                self.body3 = nn.Sequential(*features[16:23])
                self.body4 = nn.Sequential(*features[23:30])
        else:
            if name == 'vgg16_bn':
                self.body = nn.Sequential(*features[:44])  # 16x down-sample
            elif name == 'vgg16':
                self.body = nn.Sequential(*features[:30])  # 16x down-sample
        self.num_channels = num_channels
        self.return_interm_layers = return_interm_layers

    def forward(self, tensor_list):
        out = []

        if self.return_interm_layers:
            xs = tensor_list
            for _, layer in enumerate([self.body1, self.body2, self.body3, self.body4]):
                xs = layer(xs)
                out.append(xs)

        else:
            xs = self.body(tensor_list)
            out.append(xs)
        return out


class Backbone_VGG(BackboneBase_VGG):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str, return_interm_layers: bool):
        if name == 'vgg16_bn':
            backbone = models.vgg16_bn(pretrained=True)
        elif name == 'vgg16':
            backbone = models.vgg16(pretrained=True)
        num_channels = 512
        super().__init__(backbone, num_channels, name, return_interm_layers)

class BackboneBase_ResNet(nn.Module):
    def __init__(self, backbone: nn.Module, num_channels: int, name: str, return_interm_layers: bool):
        super().__init__()
        features = list(backbone.children())

        if return_interm_layers:
            # Define intermediate layers for ResNet
            self.body1 = nn.Sequential(*features[:4])  # Conv1, BN1, ReLU, MaxPool
            self.body2 = features[4]  # Layer1
            self.body3 = features[5]  # Layer2
            self.body4 = features[6]  # Layer3
        else:
            # Use the full backbone
            self.body = nn.Sequential(*features)

        self.return_interm_layers = return_interm_layers
        self.num_channels = num_channels

    def forward(self, x):
        if self.return_interm_layers:
            c1 = self.body1(x)  # Output after Conv1, BN1, ReLU, MaxPool
            c2 = self.body2(c1)  # Output of Layer1
            c3 = self.body3(c2)  # Output of Layers2
            c4 = self.body4(c3)  # Output of Layer3
            return c1, c2, c3, c4
        else:
            return self.body(x)

class Backbone_ResNet(BackboneBase_ResNet):
    def __init__(self, name: str, return_interm_layers: bool):
        if name == 'resnet50':
            backbone = resnet.resnet50(pretrained=True)
        num_channels = 1024  
        super().__init__(backbone, num_channels, name, return_interm_layers)
              
def build_backbone(args):
    if args.backbone == 'vgg16_bn':
        backbone = Backbone_VGG(args.backbone, True)
    elif args.backbone == 'resnet50':
        backbone = Backbone_ResNet(args.backbone, True)
    else:
        raise ValueError(f"Unsupported backbone type: {args.backbone}")
    return backbone



if __name__ == '__main__':
    Backbone_VGG('vgg16', True)