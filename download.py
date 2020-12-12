#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 15:59:08 2020

@author: raskshithakoriraj
"""

import os 
import urllib.request
import zipfile






#creating data folder
data_path = "data"
datasetroot = os.path.join(data_path)
datasettrain_zip = os.path.join(datasetroot,"DIV2K_train_HR.zip")
datasetvalid_zip = os.path.join(datasetroot,"DIV2K_valid_HR.zip")
train_dir = os.path.join(datasetroot,"train")
test_dir = os.path.join(datasetroot,"test")

def make_dir(data_path):
    if (os.path.exists(data_path)==False):
        os.mkdir(data_path)

def main():
    make_dir(data_path)
    make_dir(train_dir)
    make_dir(test_dir)
       
    
    
    #Downloading and extractng the tar dataset from Kaggle 
    if os.path.exists(datasetvalid_zip)==False:
        print("Downloading data for validation...")
        tar_url = 'http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip'
        filehandle, _ = urllib.request.urlretrieve(tar_url,datasetvalid_zip)
    with zipfile.ZipFile(datasetvalid_zip, 'r') as zip_ref:
        zip_ref.extractall(test_dir)
        
    
    if os.path.exists(datasettrain_zip)==False:
        print("Downloading training dataset...")
        tar_url = 'http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip'
        filehandle, _ = urllib.request.urlretrieve(tar_url,datasettrain_zip)
    with zipfile.ZipFile(datasettrain_zip, 'r') as zip_ref:
        zip_ref.extractall(train_dir)
        
if __name__ == "__main__":
    main()
