#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 23:51:14 2020

@author: rakshithakoriraj

"""
import torch.nn as nn
import torch

class DNet(nn.Module):
    def __init__(self,device = 'cpu',image_size=512):
        super().__init__()
        #self.ngpu = ngpu
        # input is 3 x image_size x image_size   output is 64 X image_size X image_size
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, 
                               padding=1)
        self.relu1 = nn.LeakyReLU(0.2)
        
        
        # input is 64 x image_size x image_size      output is 64 X image_size/2 X image_size/2
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, 
                               padding=1)
        self.batchnorm2 = nn.BatchNorm2d(64)
        self.relu2 = nn.LeakyReLU(0.2)
        
        #------------#        
        # input is 64 X image_size/2 X image_size/2      output is 128 X image_size/4 X image_size/4
        self.conv3 = convnet(feature_map=64).to(device) #feature_map=64
        
        
        # input is 128 X image_size/4 X image_size/4      output is 256 X image_size/8 X image_size/8
        self.conv4 = convnet(feature_map=128).to(device) #feature_map=128
        
        
        # input is 256 X image_size/8 X image_size/8      output is 512 X image_size/16 X image_size/16
        self.conv5 = convnet(feature_map=256).to(device) #feature_map=256
        """
        self.F1 = nn.Flatten()
        self.linear1 = nn.Linear(2 * image_size * image_size, 1024)
        self.lRelu = nn.LeakyReLU(0.2)
        self.linear2 = nn.Linear(1024, 1)
        self.sigmoid = nn.Sigmoid()
        """
        self.aap1 = nn.AdaptiveAvgPool2d(1)
        self.conv6 = nn.Conv2d(512, 1024, kernel_size=1)
        self.llx = nn.LeakyReLU(0.2)
        self.conv7 =  nn.Conv2d(1024, 1, kernel_size=1)
    def forward(self, input):
        batch_size = input.size(0)
        x = self.conv1(input)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        '''
        x = self.F1(x) #flatten
        x = self.linear1(x)
        x = self.lRelu(x)
        x = self.linear2(x)
        x = self.sigmoid(x)
        '''
        x = self.aap1(x)
        x = self.conv6(x)
        x = self.llx(x)
        x = self.conv7(x)
        x = torch.sigmoid(x.view(batch_size))
        return x        
    
    
        
        
        
class convnet(nn.Module):
    def __init__(self,feature_map=64):
        super().__init__() #
        self.main = nn.Sequential(
            nn.Conv2d(in_channels=feature_map, out_channels=feature_map*2, kernel_size=3, 
                      stride=1, padding=1),
            nn.BatchNorm2d(feature_map*2),
            nn.LeakyReLU(0.2),
            
            
            
            nn.Conv2d(in_channels=feature_map*2, out_channels=feature_map*2, kernel_size=3, 
                      stride=2, padding=1),
            nn.BatchNorm2d(feature_map*2),
            nn.LeakyReLU(0.2)        
            
            
            )

    def forward(self, input):
        output = self.main(input)
        return output
        

    
    
    
        