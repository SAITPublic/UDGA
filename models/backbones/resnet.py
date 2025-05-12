# Copyright (c) Phigent Robotics. All rights reserved.

import torch
from torch import nn

from mmdet.models.backbones.resnet import Bottleneck, BasicBlock, ResNet
import torch.utils.checkpoint as checkpoint

from mmdet.models import BACKBONES

from mmdet3d.models.necks import TransformerBlock
from mmdet3d.models.utils import peft_conv2d, peft_bn2d, peft_convmodule
from mmdet3d.models.utils import Adapter


@BACKBONES.register_module()
class Adapter_ResNet(ResNet):
    def __init__(self, **kwargs):
        super(Adapter_ResNet, self).__init__(**kwargs)

        self.adapter_input = nn.Sequential(nn.BatchNorm2d(3),
                                       Adapter(3, 64, 1),
                                       nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True),
                                       Adapter(64, 64, 1),)
                                       
        for param in self.conv1.parameters():
            param.requries_grad = False
        for param in self.norm1.parameters():
            param.requries_grad = False
        for param in self.relu.parameters():
            param.requries_grad = False
        for param in self.maxpool.parameters():
            param.requries_grad = False
        
        self.adapter_1 = Adapter(64, 64, 2)
        self.adapter_2 = nn.Sequential(nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True),
                                       Adapter(64, 64, 1),)
        mul_feats = [256,512,1024,2048]
        self.adapters_layers = []
        in_ch = 64
        for out_ch in mul_feats:
            lays = []
            if in_ch == 64:
                lays.append(Adapter(in_ch, out_ch, 2))
            else:
                lays.append(nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True))
                lays.append(Adapter(in_ch, out_ch, 4))
            in_ch = out_ch
            self.adapters_layers.append(nn.Sequential(*lays).cuda())

        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            for param in res_layer.parameters():
                param.requires_grad  = False

    def forward(self, x):
        """Forward function."""
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x) + self.adapter_input(x)
            x = self.norm1(x)
            x = self.relu(x) + self.adapter_1(x)

        x = self.maxpool(x) + self.adapter_2(x)
        
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x) + self.adapters_layers[i](x)
            if i in self.out_indices:
                outs.append(x)
    
        return tuple(outs)

        # BEFORE 0 : torch.Size([48, 64, 64, 176])
        # AFTER 0 : torch.Size([48, 256, 64, 176])
        # BEFORE 1 : torch.Size([48, 256, 64, 176])
        # AFTER 1 : torch.Size([48, 512, 32, 88])
        # BEFORE 2 : torch.Size([48, 512, 32, 88])
        # AFTER 2 : torch.Size([48, 1024, 16, 44])
        # BEFORE 3 : torch.Size([48, 1024, 16, 44])
        # AFTER 3 : torch.Size([48, 2048, 8, 22])
    


@BACKBONES.register_module()
class ResNetForBEVDet(nn.Module):
    def __init__(self, numC_input, num_layer=[2,2,2], num_channels=None, stride=[2,2,2],
                 backbone_output_ids=None, norm_cfg=dict(type='BN'),
                 with_cp=False, block_type='Basic',):
        super(ResNetForBEVDet, self).__init__()
        #build backbone
        # assert len(num_layer)>=3
        assert len(num_layer)==len(stride)
        num_channels = [numC_input*2**(i+1) for i in range(len(num_layer))] \
            if num_channels is None else num_channels
        self.backbone_output_ids = range(len(num_layer)) \
            if backbone_output_ids is None else backbone_output_ids
        layers = []
        if block_type == 'BottleNeck':
            curr_numC = numC_input
            for i in range(len(num_layer)):
                layer=[Bottleneck(curr_numC, num_channels[i]//4, stride=stride[i],
                                  downsample=nn.Conv2d(curr_numC,num_channels[i],3,stride[i],1),
                                  norm_cfg=norm_cfg)]
                curr_numC= num_channels[i]
                layer.extend([Bottleneck(curr_numC, curr_numC//4,
                                         norm_cfg=norm_cfg) for _ in range(num_layer[i]-1)])
                layers.append(nn.Sequential(*layer))
        elif block_type == 'Basic':
            curr_numC = numC_input
            for i in range(len(num_layer)):
                layer=[BasicBlock(curr_numC, num_channels[i], stride=stride[i],
                                  downsample=nn.Conv2d(curr_numC,num_channels[i],3,stride[i],1),
                                  norm_cfg=norm_cfg)]
                curr_numC= num_channels[i]
                layer.extend([BasicBlock(curr_numC, curr_numC, norm_cfg=norm_cfg) for _ in range(num_layer[i]-1)])
                layers.append(nn.Sequential(*layer))
        else:
            assert False
        self.layers = nn.Sequential(*layers)

        self.with_cp = with_cp

    def forward(self, x):
        feats = []
        x_tmp = x
        for lid, layer in enumerate(self.layers):
            if self.with_cp:
                x_tmp = checkpoint.checkpoint(layer, x_tmp)
            else:
                x_tmp = layer(x_tmp)
            if lid in self.backbone_output_ids:
                feats.append(x_tmp)
        return feats

def init_ssf_scale_shift(dim):
    scale = nn.Parameter(torch.ones(dim)).cuda()
    shift = nn.Parameter(torch.zeros(dim)).cuda()

    nn.init.normal_(scale, mean=1, std=.02)
    nn.init.normal_(shift, std=.02)

    return scale, shift


def ssf_ada(x, scale, shift):
    assert scale.shape == shift.shape
    if x.shape[-1] == scale.shape[0]:
        return x * scale + shift
    elif x.shape[1] == scale.shape[0]:
        return x * scale.view(1, -1, 1, 1) + shift.view(1, -1, 1, 1)
    else:
        raise ValueError('the input tensor shape does not match the shape of the scale factor.')


@BACKBONES.register_module()
class PEFTResNetForBEVDet(nn.Module):
    def __init__(self, numC_input, num_layer=[2,2,2], num_channels=None, stride=[2,2,2],
                 backbone_output_ids=None, norm_cfg=dict(type='BN'),
                 with_cp=False, block_type='Basic',):
        super(PEFTResNetForBEVDet, self).__init__()
        #build backbone
        # assert len(num_layer)>=3
        assert len(num_layer)==len(stride)
        num_channels = [numC_input*2**(i+1) for i in range(len(num_layer))] \
            if num_channels is None else num_channels
        self.backbone_output_ids = range(len(num_layer)) \
            if backbone_output_ids is None else backbone_output_ids
        layers = []
        self.ssfs = []
        if block_type == 'BottleNeck':
            curr_numC = numC_input
            for i in range(len(num_layer)):
                layer=[Bottleneck(curr_numC, num_channels[i]//4, stride=stride[i],
                                  downsample=nn.Conv2d(curr_numC,num_channels[i],3,stride[i],1),
                                  norm_cfg=norm_cfg)]
                curr_numC= num_channels[i]
                layer.extend([Bottleneck(curr_numC, curr_numC//4,
                                         norm_cfg=norm_cfg) for _ in range(num_layer[i]-1)])
                layers.append(nn.Sequential(*layer))
                scale, shift = init_ssf_scale_shift(curr_numC//4)
                self.ssfs.append([scale, shift])
        elif block_type == 'Basic':
            curr_numC = numC_input
            for i in range(len(num_layer)):
                layer=[BasicBlock(curr_numC, num_channels[i], stride=stride[i],
                                  downsample=nn.Conv2d(curr_numC,num_channels[i],3,stride[i],1),
                                  norm_cfg=norm_cfg)]
                curr_numC= num_channels[i]
                layer.extend([BasicBlock(curr_numC, curr_numC, norm_cfg=norm_cfg) for _ in range(num_layer[i]-1)])
                layers.append(nn.Sequential(*layer))
                scale, shift = init_ssf_scale_shift(curr_numC)
                self.ssfs.append([scale, shift])
        else:
            assert False
        self.layers = nn.Sequential(*layers)

        for param in self.layers.parameters():
            param.requires_grad = False

        self.with_cp = with_cp

    def forward(self, x):
        feats = []
        x_tmp = x
        for lid, layer in enumerate(self.layers):
            if self.with_cp:
                x_tmp = checkpoint.checkpoint(layer, x_tmp)
            else:
                x_tmp = layer(x_tmp)
            x_tmp = ssf_ada(x_tmp, self.ssfs[lid][0], self.ssfs[lid][1])
            if lid in self.backbone_output_ids:
                feats.append(x_tmp)
        return feats



@BACKBONES.register_module()
class AdaptResNetForBEVDet(nn.Module):
    def __init__(self, numC_input, num_layer=[2,2,2], num_channels=None, stride=[2,2,2],
                 backbone_output_ids=None, norm_cfg=dict(type='BN'),
                 with_cp=False, block_type='Basic',):
        super(AdaptResNetForBEVDet, self).__init__()
        #build backbone
        # assert len(num_layer)>=3
        assert len(num_layer)==len(stride)
        self.dep = True
        if len(num_layer) > 1: self.dep = False 

        num_channels = [numC_input*2**(i+1) for i in range(len(num_layer))] \
            if num_channels is None else num_channels
        self.backbone_output_ids = range(len(num_layer)) \
            if backbone_output_ids is None else backbone_output_ids
        layers = []
        adapters = []
        if block_type == 'BottleNeck':
            curr_numC = numC_input
            for i in range(len(num_layer)):
                layer=[Bottleneck(curr_numC, num_channels[i]//4, stride=stride[i],
                                  downsample=nn.Conv2d(curr_numC,num_channels[i],3,stride[i],1),
                                  norm_cfg=norm_cfg)]       
                curr_numC= num_channels[i]
                layer.extend([Bottleneck(curr_numC, curr_numC//4,
                                         norm_cfg=norm_cfg) for _ in range(num_layer[i]-1)])
                layers.append(nn.Sequential(*layer))
        elif block_type == 'Basic':
            curr_numC = numC_input
            for i in range(len(num_layer)):
                if self.dep == True:
                    self.adapter = Adapter(curr_numC, curr_numC)
                else:
                    apt=[Adapter(curr_numC, num_channels[i])]
                layer=[BasicBlock(curr_numC, num_channels[i], stride=stride[i],
                                  downsample=nn.Conv2d(curr_numC,num_channels[i],3,stride[i],1),
                                  norm_cfg=norm_cfg)]
                curr_numC= num_channels[i]
                layer.extend([BasicBlock(curr_numC, curr_numC, norm_cfg=norm_cfg) for _ in range(num_layer[i]-1)])
                layers.append(nn.Sequential(*layer))   
                if not self.dep:
                    apt.extend([Adapter(curr_numC,curr_numC) for _ in range(num_layer[i]-1)])
                    adapters.append(nn.Sequential(*apt))
        else:
            assert False
        self.layers = nn.Sequential(*layers)
        self.downsample = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)
        self.adapters = nn.Sequential(*adapters)


        for param in self.layers.parameters():
            param.requires_grad = False

        self.with_cp = with_cp

    def forward(self, x):
        feats = []
        x_tmp = x
        if self.dep:
            for lid, layer in enumerate(self.layers):
                if self.with_cp:
                    x_tmp = checkpoint.checkpoint(layer, x_tmp)
                else:
                    x_tmp = layer(x_tmp) + self.adapter(x_tmp)
                if lid in self.backbone_output_ids:
                    feats.append(x_tmp)
        else:
            for lid, layer in enumerate(self.layers):
                if self.with_cp:
                    x_tmp = checkpoint.checkpoint(layer, x_tmp)
                else:
                    x_tmp = layer(x_tmp) + self.adapters[lid](self.downsample(x_tmp))
                if lid in self.backbone_output_ids:
                    feats.append(x_tmp)
        return feats