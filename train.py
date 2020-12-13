#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 23:53:14 2020

@author: rakshithakoriraj
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


EPOCHS = 1
batch_size = 5
hr_image_size = 512
workers = 4
lr_image_size = 128
testbatchsize = 32


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

realPath = os.path.join(os.path.abspath(os.getcwd()),'data','train')
testPath = os.path.join(os.path.abspath(os.getcwd()),'data', 'test')
genImagesPath = os.path.abspath(os.path.join(os.getcwd(), "data","genImagesE{}-B{}".format(EPOCHS, batch_size)))
make_folder(genImagesPath)
SRImagesPath = os.path.abspath(os.path.join(genImagesPath, "SRImages"))
make_folder(SRImagesPath)



# Create the dataset
dataset = dset.ImageFolder(root=realPath,
                           transform=transforms.Compose([
                               transforms.Resize(hr_image_size),
                               transforms.CenterCrop(hr_image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
# Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)


#-------------------------------------#      
#Discriminator
discriminator = SRDiscriminator.DNet(device,hr_image_size).to(device)
#discriminator = SRDiscriminator.Discriminator().to(device)
print("discriminator paramenters {}".format(discriminator))
# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    discriminator = nn.DataParallel(discriminator, list(range(ngpu)))
    
#-------------------------------------#
#Generator
#Generating images from noise
generator = SRGenerator.GNet(device).to(device)
print("generator paramenters {}".format(generator))
# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    generator = nn.DataParallel(generator, list(range(ngpu)))
    
datasetTest = dset.ImageFolder(root=testPath,
                           transform=transforms.Compose([
                               transforms.Resize(lr_image_size),
                               transforms.CenterCrop(lr_image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
# Create the dataloader
dataloaderTest = torch.utils.data.DataLoader(datasetTest, batch_size=testbatchsize,
                                         shuffle=False, num_workers=1)



loss_fuction_BCE = nn.BCELoss()
loss_fuction_MSE = nn.MSELoss()
gOptimizer = optim.Adam(generator.parameters(),lr=0.0002, betas=(0.5, 0.999))
dOptimizer = optim.Adam(discriminator.parameters(),lr=0.0002, betas=(0.5, 0.999))

errG = []
errD = []
image_loss_list = []
adversarial_loss_list = []
perception_loss_list = []


VGGnet = models.vgg16_bn(pretrained=True).features.to(device)
VGGnet.eval()
for param in VGGnet.parameters():
    param.requires_grad = False

#-------------------------------------#                
print("Pre training the Generator ")
low_res = []
for epoch in range(2):
    mean_generator_content_loss = 0.0

    for i, data in enumerate(dataloader):
        #if i > 5:
         #   break
        
        # Downsample images to low resolution
        for j in range(batch_size):
            highOriginal = data[0].to(device)
            lowOriginal = F.interpolate(highOriginal, size=(lr_image_size, lr_image_size), mode='bicubic')

        ######### Train generator #########
        generator.zero_grad()

        highgen = generator(lowOriginal) 
        image_loss = loss_fuction_MSE(highgen, highOriginal)
        image_loss.backward()
        gOptimizer.step()
        print("Pre-trained Generator: batch:{},epoch:{},MSELoss{}".format(i,epoch,image_loss))

    
    
#-------------------------------------#                
for epoch in range(EPOCHS):
    #for batch in range(0, len(dimageList), BATCH):
    for i, data in enumerate(dataloader):
        #if i > 5:
         #   break
        highOriginal = data[0].to(device)
        lowOriginal = F.interpolate(highOriginal, size=(lr_image_size, lr_image_size), mode='bicubic')

        
        #Generator training
        highgen = generator(lowOriginal) 
        
        
        real_labels = torch.ones(len(highOriginal), device=device)
        fake_labels = torch.zeros(len(lowOriginal), device=device)
        
        #Discriminator training
        discriminator.zero_grad()
        highOriginalY = discriminator(highOriginal).view(-1)
        highgenY = discriminator(highgen.detach()).view(-1)
        
        
        highDloss = loss_fuction_BCE(highOriginalY, real_labels)
        D_x = highDloss.mean().item()
        highGloss = loss_fuction_BCE(highgenY, fake_labels)
        G_x = highGloss.mean().item()
        
        
        totalLoss = (highDloss + highGloss)
        totalLoss.backward()
        dOptimizer.step()
        errD.append(totalLoss)

        
        
        generator.zero_grad()
        with torch.no_grad():
            vgg_highgenfeature = VGGnet(highgen.detach())
            vgg_highfeature = VGGnet(highOriginal)
            perception_loss = loss_fuction_MSE(vgg_highgenfeature, vgg_highfeature)
        highgenY_1 = discriminator(highgen.detach()).view(-1)        
        image_loss = loss_fuction_MSE(highgen, highOriginal)
        adversarial_loss = loss_fuction_BCE(highgenY_1, real_labels)
        totalGloss = (image_loss+0.001*adversarial_loss+0.006*perception_loss)
        totalGloss.backward()
        gOptimizer.step()


        
        
        #saving losses to plot grahs
        errG.append(totalGloss)
        image_loss_list.append(image_loss)
        adversarial_loss_list.append(adversarial_loss)
        perception_loss_list.append(perception_loss)
        
        print("batch:{},epoch:{},totalLoss{},totalGloss{}".format(i,epoch,totalLoss,totalGloss))

        
        '''
        if i %5== 0:
            #print("totalLoss{},los{}".format(totalLoss,los))
            with torch.no_grad():
                gen_batch = next(iter(datasetTest))
                fig = plt.figure(figsize=(8,8))
                plt.axis("off")
                plt.title("test Images")
                plt.imshow(np.transpose(vutils.make_grid(gen_batch[0][0:64], padding=5, normalize=True, pad_value = 1),(1,2,0)))
                fig.savefig(os.path.join(SRImagesPath,'{}-trainimage-ts{}.png'.format(i,int(time.time()))))
                genbatch = gen_batch[0].to(device)
                highgen = generator(genbatch)
                fig = plt.figure(figsize=(8,8))
                plt.axis("off")
                plt.title("Generated Images")
                plt.imshow(np.transpose(vutils.make_grid(highgen.cpu()[0:64], padding=5, normalize=True, pad_value = 1),(1,2,0)))
                fig.savefig(os.path.join(SRImagesPath,'{}-trainSRimage-ts{}.png'.format(i,int(time.time()))))
        
       '''



fig = plt.figure(figsize=(20,20))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(errG,label="G")
plt.plot(errD,label="D")
#plt.plot(errD,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
fig.savefig(os.path.join(SRImagesPath,'graph-ts{}.png'.format(int(time.time()))))


fig = plt.figure(figsize=(20,20))
plt.title("losses")
plt.plot(image_loss_list,label="IL")
plt.plot(adversarial_loss_list,label="AL")
plt.plot(perception_loss_list,label="PL")
plt.plot(errG,label="G")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
fig.savefig(os.path.join(SRImagesPath,'gen_graph-losses{}.png'.format(int(time.time()))))

make_folder(os.path.join(os.path.abspath(os.getcwd()),'models'))
torch.save(generator.state_dict(), 
           os.path.join(os.path.abspath(os.getcwd()),'models',"state_dict_model{}.pt".format(int(time.time()))))




with torch.no_grad():
    for i, data in enumerate(dataloaderTest):
        test_batch = data[0].to(device)
        for j, lr_image in enumerate(test_batch):
            vutils.save_image(lr_image,os.path.join(SRImagesPath,'{}-LRimage-ts{}.png'.format(j,int(time.time()))),normalize=True)
        highgen = generator(test_batch)
        for j, sr_image in enumerate(highgen.cpu()):
            vutils.save_image(sr_image,os.path.join(SRImagesPath,'{}-SRimage-ts{}.png'.format(j,int(time.time()))),normalize=True)
    
    
    
