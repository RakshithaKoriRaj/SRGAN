#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 18:41:07 2020

@author: raskshithakoriraj
"""



#libraries
import time
import numpy as np
from matplotlib import pyplot as plt
import torch.nn.functional as F
import os
import SRGenerator
import SRDiscriminator
import torchvision.models as models
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import argparse
import config as cg

EPOCHS = cg.EPOCHS
hr_image_size = cg.hr_image_size
workers = cg.workers
lr_image_size = cg.lr_image_size
testbatchsize = cg.testbatchsize

parser = argparse.ArgumentParser()
gen_model_path = os.path.join(os.path.abspath(os.getcwd()),'models',"generator_model_lr_{}_hr_{}.pt".format(lr_image_size,hr_image_size))
parser.add_argument('--generatorWeights', type=str, default=gen_model_path, help="path to generator weights (to continue training)")


opt = parser.parse_args()
print(opt)




# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

# Decide on which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

def make_folder(path):
	try:
		os.mkdir(path)
	except FileNotFoundError:
		print("{} path can't be created".format(path))
	except FileExistsError:
		print("{} folder already exists".format(path))

#realPath = os.path.join(os.path.abspath(os.getcwd()),'data','train')
testPath = os.path.join(os.path.abspath(os.getcwd()),'data', 'test')
genImagesPath = os.path.abspath(os.path.join(os.getcwd(), "data","genImagesE{}-B{}".format(EPOCHS, testbatchsize)))
make_folder(genImagesPath)
SRImagesPath = os.path.abspath(os.path.join(genImagesPath, "SRImages"))
make_folder(SRImagesPath)



# Create the dataset
dataset = dset.ImageFolder(root=testPath,
                           transform=transforms.Compose([
                               transforms.Resize(hr_image_size),
                               transforms.CenterCrop(hr_image_size),
                               transforms.ToTensor(),
                               #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
# Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=testbatchsize,
                                         shuffle=True, num_workers=workers)
   
#-------------------------------------#
#Generator
#Generating images from noise
generator = SRGenerator.GNet(device).to(device)
print("generator paramenters {}".format(generator))
# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    generator = nn.DataParallel(generator, list(range(ngpu)))
    
    
normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    
scale = transforms.Compose([
                               transforms.ToPILImage(),
                               transforms.Resize(lr_image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ])

loss_fuction_BCE = nn.BCELoss()
loss_fuction_MSE = nn.MSELoss()

generator.load_state_dict(torch.load(opt.generatorWeights))
generator.eval()

image_loss = 0.0
perception_loss = 0.0
totalGloss = 0.0

VGGnet = models.vgg16_bn(pretrained=True).features.to(device)
VGGnet.eval()
for param in VGGnet.parameters():
    param.requires_grad = False

#-------------------------------------#                
#for batch in range(0, len(dimageList), BATCH):
for i, data in enumerate(dataloader):
    if i > 5:
        break
    highOriginal = data[0].to(device)
    #lowOriginal = F.interpolate(highOriginal, size=(lr_image_size, lr_image_size), mode='bicubic')
    lowOriginal = torch.FloatTensor(testbatchsize, 3, lr_image_size, lr_image_size)
    for j in range(testbatchsize):
        lowOriginal[j] = scale(highOriginal[j])
        highOriginal[j]=normalize(highOriginal[j])
    
    #Generator training
    highgen = generator(lowOriginal) 
    
    
    vgg_highgenfeature = VGGnet(highgen.detach())
    vgg_highfeature = VGGnet(highOriginal)
    perception_loss += loss_fuction_MSE(vgg_highgenfeature, vgg_highfeature)
    image_loss += loss_fuction_MSE(highgen, highOriginal)
    totalGloss += (image_loss+0.006*perception_loss)
    
    for j, hr_image in enumerate(highOriginal):
        vutils.save_image(hr_image,os.path.join(SRImagesPath,'{}-{}-HRimage-ts{}.png'.format(i,j,int(time.time()))),normalize=True)

    for j, lr_image in enumerate(lowOriginal):
        vutils.save_image(lr_image,os.path.join(SRImagesPath,'{}-{}-LRimage-ts{}.png'.format(i,j,int(time.time()))),normalize=True)

    for j, sr_image in enumerate(highgen.cpu()):
        vutils.save_image(sr_image,os.path.join(SRImagesPath,'{}-{}-SRimage-ts{}.png'.format(i,j,int(time.time()))),normalize=True)

        
totalGloss = totalGloss/len(dataloader)
image_loss = image_loss/len(dataloader)
perception_loss = perception_loss/len(dataloader)


print("Generator loss {}".format(totalGloss))
print("image loss {}".format(image_loss))  
print("perception loss {}".format(perception_loss))