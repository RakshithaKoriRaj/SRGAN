#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 10:44:22 2020

@author: rakshithakoriraj
Data filtering
"""

import os
import shutil
import cv2


stdSize = (128,128)

# Creates a folder at path location
# does not recursively create path
def make_folder(path):
	try:
		os.mkdir(path)
	except FileNotFoundError:
		print("{} path can't be created".format(path))
	except FileExistsError:
		print("{} folder already exists".format(path))
        
        
def copy_and_filter_dataset(src, dst):
    for item in os.listdir(src):
        #if "png" in item.lower():
        s = os.path.join(src, item)
        shutil.copy(s, dst)


def rename_all(src, dst):
    count = 0
    for item in os.listdir(src):
        try: 
            filename, file_extension = os.path.splitext(os.path.join(src, item))
            shutil.copy(os.path.join(src, item), dst)
            os.rename(os.path.join(dst,item), os.path.join(dst,str(count))+file_extension)
            count += 1
        except FileExistsError:
            print("Already renamed {}".format(item))
            

def sample_Dataset(src, dst):
    for item in os.listdir(src):
        try:
            fullpath = os.path.join(src, item)
            image = cv2.imread(fullpath)
            #resizing to 128X128
            output = cv2.resize(image, stdSize, interpolation=cv2.INTER_LINEAR)
            filename, fileextension = os.path.splitext(item)
            wrtePath = os.path.join(dst, item)
            if not cv2.imwrite(wrtePath, output):
                print("Write failed")       
        except Exception as e:
            print(fullpath)
            pass
       
    
if __name__ == "__main__":
    
    #creating seperate folders for different kinfds of dataset
    dir1 = os.path.join(os.path.abspath(os.getcwd()),"DIV2K_train_HR")
    
    '''
    base = os.path.abspath(os.path.join(os.getcwd(), "datasets"))
    make_folder(base)
    print("base----->"+base)
    
    
    nameLR = os.path.join(base, "nameLR")
    make_folder(nameLR)
    print("Low resolution training dataset----->"+nameLR)
    copy_and_filter_dataset(dir1, nameLR)
    '''
    
    #sampling- upsampling or downsampling
    LRimages = os.path.join(os.path.abspath(os.getcwd()), "LRimages")
    make_folder(LRimages)
    print("LRimages images----->"+LRimages)
    sample_Dataset(dir1, LRimages)

    