#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 23:51:14 2020

@author: rakshithakoriraj
"""

import torch.nn as nn
import torch.nn.functional as F

class GNet(nn.Module):
    def __init__(self,device = 'cpu',scale_factor=2):
        super().__init__()
        #in-128x128x3  out-128x128x64
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=9, 
                      stride=1, padding=4)
        self.prelu1 = nn.PReLU()
        
        #in-128x128x64  out-128x128x64
        self.resnet1 = ResNet().to(device)
            
        self.resnet2 = ResNet().to(device)
        self.batchnorm = nn.BatchNorm2d(64)
        
        #in-128x128x64   out-128x128x64
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, 
                      stride=1, padding=1)
        
        self.scalenets = []
        
        for scale in range(scale_factor):
            scalenet = ScaleNet(ic=64,scale=2).to(device)
            self.scalenets.append(scalenet)
            
        '''   
        #in-128x128x64   out-256x256x64
        self.scalenet1 = ScaleNet(ic=64,scale=2)
        self.scalenets.append(self.scalenet1)
        #in-256x256x64   out-512x512x64
        self.scalenet2 = ScaleNet(ic=64,scale=2)
        self.scalenets.append(self.scalenet2)
        '''
        
        #in-512x512x64   out-512x512x3
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=9, 
                      stride=1, padding=4)
        
          
        
    def forward(self, input):
        x = self.conv1(input)
        x = self.prelu1(x)
        x1 = x
        x = self.resnet1(x)
        x = self.resnet2(x)
        x = self.batchnorm(self.conv2(x))
        x = x+x1
        for scalenet in self.scalenets:
            x = scalenet(x)
        #x = self.scalenet1(x)
        #x = self.scalenet2(x)
        x = self.conv3(x)
        return x

            
    
    
    
    
class ResNet(nn.Module):
    def __init__(self,feature_map=64):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels=feature_map, out_channels=feature_map, kernel_size=3, 
                      stride=1, padding=1),
            nn.BatchNorm2d(feature_map),
            nn.PReLU(),
            
            
            nn.Conv2d(in_channels=feature_map, out_channels=feature_map, kernel_size=3, 
                      stride=1, padding=1),
            nn.BatchNorm2d(feature_map)
            
            )
        
    def forward(self, input):
        output = input + self.main(input)
        return output


class ScaleNet(nn.Module):
    def __init__(self,ic=64,scale=2):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels=ic, out_channels=ic*(scale**2), kernel_size=3, 
                      stride=1, padding=1),
            nn.PixelShuffle(scale), #in=out_channels   out=out_channels/scale**2    imgsize=imagesize*(Xscale) 
            nn.PReLU(),
            )
    
    def forward(self, input):
        return self.main(input)
    
