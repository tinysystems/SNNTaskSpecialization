from __future__ import print_function
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from torchvision import datasets, transforms
from model import *
from dataset import NMNIST

import sys






device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(device)


batch_size = 16





### Create the outputs and save them 

create_outputs = 0

if create_outputs == 1: 

    ## Load the trained models; sub-tasks 
    num_subTasks = 10 
    subTask_model = [] # subSNN

    scale_factors = []
    for i in range(10):
        scale_factors.append(0)

    # Scale factor 16
    scale_factors[0] = 16
    scale_factors[1] = 16
    scale_factors[6] = 16
    scale_factors[7] = 16
    scale_factors[8] = 16
    scale_factors[9] = 16

    # Scale factor 8
    scale_factors[4] = 8
    scale_factors[5] = 8

    # Scale factor 4 but for practical reasons trained with 8 (to speed up the training process)
    scale_factors[2] = 8
    scale_factors[3] = 8

    ## Models' paths

    MODEL_PATH = []
    for i in range(10):
        MODEL_PATH.append('')


    MODEL_PATH[0] = f'/home/muath.abulebdeh/experiments/project13/savings/variables/cifar10_vgg9_scale_factor{scale_factors[0]}/'
    MODEL_PATH[0] = MODEL_PATH[0] + 'cifar10_vgg9_subSNN_0_acc_91.5_scale_factor16_epoch164.pt'
    MODEL_PATH[1] = f'/home/muath.abulebdeh/experiments/project13/savings/variables/cifar10_vgg9_scale_factor{scale_factors[1]}/'
    MODEL_PATH[1] = MODEL_PATH[1] + 'cifar10_vgg9_subSNN_1_acc_94.29_scale_factor16_epoch73.pt'
    MODEL_PATH[2] = f'/home/muath.abulebdeh/experiments/project13/savings/variables/cifar10_vgg9_scale_factor{scale_factors[2]}/'
    MODEL_PATH[2] = MODEL_PATH[2] + 'cifar10_vgg9_subSNN_2_acc_87.91_scale_factor8_epoch128.pt'
    MODEL_PATH[3] = f'/home/muath.abulebdeh/experiments/project13/savings/variables/cifar10_vgg9_scale_factor{scale_factors[3]}/'
    MODEL_PATH[3] = MODEL_PATH[3] + 'cifar10_vgg9_subSNN_3_acc_83_scale_factor8_epoch170_lowAccuracy.pt'
    MODEL_PATH[4] = f'/home/muath.abulebdeh/experiments/project13/savings/variables/cifar10_vgg9_scale_factor{scale_factors[4]}/'
    MODEL_PATH[4] = MODEL_PATH[4] + 'cifar10_vgg9_subSNN_4_acc_89.98_scale_factor8_epoch112.pt'
    MODEL_PATH[5] = f'/home/muath.abulebdeh/experiments/project13/savings/variables/cifar10_vgg9_scale_factor{scale_factors[5]}/'
    MODEL_PATH[5] = MODEL_PATH[5] + 'cifar10_vgg9_subSNN_5_acc_88.55_scale_factor8_epoch122.pt'
    MODEL_PATH[6] = f'/home/muath.abulebdeh/experiments/project13/savings/variables/cifar10_vgg9_scale_factor{scale_factors[6]}/'
    MODEL_PATH[6] = MODEL_PATH[6] + 'cifar10_vgg9_subSNN_6_acc_92.3_scale_factor16_epoch170.pt'
    MODEL_PATH[7] = f'/home/muath.abulebdeh/experiments/project13/savings/variables/cifar10_vgg9_scale_factor{scale_factors[7]}/'
    MODEL_PATH[7] = MODEL_PATH[7] + 'cifar10_vgg9_subSNN_7_acc_89.12_scale_factor16_epoch104.pt'
    MODEL_PATH[8] = f'/home/muath.abulebdeh/experiments/project13/savings/variables/cifar10_vgg9_scale_factor{scale_factors[8]}/'
    MODEL_PATH[8] = MODEL_PATH[8] + 'cifar10_vgg9_subSNN_8_acc_91.74_scale_factor16_epoch42.pt'
    MODEL_PATH[9] = f'/home/muath.abulebdeh/experiments/project13/savings/variables/cifar10_vgg9_scale_factor{scale_factors[9]}/'
    MODEL_PATH[9] = MODEL_PATH[9] + 'cifar10_vgg9_subSNN_9_acc_91.28_scale_factor16_epoch105.pt'


    # Load the modols for sub-tasks 

    # subSNN 0
    subTask_model.append(sub_VGG9(scale_factor=scale_factors[0]).to(device))
    subTask_model[0].load_state_dict(torch.load(MODEL_PATH[0], map_location=device))
    subTask_model[0].eval

    # subSNN 1
    subTask_model.append(sub_VGG9(scale_factor=scale_factors[1]).to(device))
    subTask_model[1].load_state_dict(torch.load(MODEL_PATH[1], map_location=device))
    subTask_model[1].eval


    # subSNN 2
    subTask_model.append(sub_VGG9(scale_factor=scale_factors[2]).to(device))
    subTask_model[2].load_state_dict(torch.load(MODEL_PATH[2], map_location=device))
    subTask_model[2].eval


    # subSNN 3
    subTask_model.append(sub_VGG9(scale_factor=scale_factors[3]).to(device))
    subTask_model[3].load_state_dict(torch.load(MODEL_PATH[3], map_location=device))
    subTask_model[3].eval


    # subSNN 4
    subTask_model.append(sub_VGG9(scale_factor=scale_factors[4]).to(device))
    subTask_model[4].load_state_dict(torch.load(MODEL_PATH[4], map_location=device))
    subTask_model[4].eval


    # subSNN 5
    subTask_model.append(sub_VGG9(scale_factor=scale_factors[5]).to(device))
    subTask_model[5].load_state_dict(torch.load(MODEL_PATH[5], map_location=device))
    subTask_model[5].eval


    # subSNN 6
    subTask_model.append(sub_VGG9(scale_factor=scale_factors[6]).to(device))
    subTask_model[6].load_state_dict(torch.load(MODEL_PATH[6], map_location=device))
    subTask_model[6].eval


    # subSNN 7
    subTask_model.append(sub_VGG9(scale_factor=scale_factors[7]).to(device))
    subTask_model[7].load_state_dict(torch.load(MODEL_PATH[7], map_location=device))
    subTask_model[7].eval


    # subSNN 8
    subTask_model.append(sub_VGG9(scale_factor=scale_factors[8]).to(device))
    subTask_model[8].load_state_dict(torch.load(MODEL_PATH[8], map_location=device))
    subTask_model[8].eval


    # subSNN 9
    subTask_model.append(sub_VGG9(scale_factor=scale_factors[9]).to(device))
    subTask_model[9].load_state_dict(torch.load(MODEL_PATH[9], map_location=device))
    subTask_model[9].eval


    MODEL_PATH = 0
    scale_factors = 0




    ## Load the dataset 

    DATA_PATH = '/home/muath.abulebdeh/colab_projects/data/cifar10/'

    train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10(DATA_PATH, train=True, download=True,
                    transform=transforms.Compose([
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomCrop(32, padding=4),
                        transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
                    ])),
    batch_size=batch_size, shuffle=True)

    output_set = [] 
    for batch_idx, (data, target) in enumerate(train_loader):

        data, target = data.to(device), target.to(device)
        data, _ = torch.broadcast_tensors(data, torch.zeros((steps,) + data.shape)) 
        data = data.permute(1, 2, 3, 4, 0)

        intermediate_output = []
        for model_index in range(num_subTasks):
            out, _, _ = subTask_model[model_index](data) 
            intermediate_output.append(out.mean(dim=2)) 

        intermediate_output = torch.squeeze(torch.stack(intermediate_output, dim=1))

        
        output_set.append(intermediate_output)
        output_set.append(target)



        SAVING_PATH = f'/home/muath.abulebdeh/experiments/project13/savings/variables/preprossed_output_data/output_batch{batch_idx}.pt'
        torch.save(output_set, SAVING_PATH)
        output_set = [] 
        print(batch_idx)




### Create a single data file / datsaet from all output batches

create_dataset = 1-create_outputs

if create_dataset == 1: 
    
    preprocessed_outputs_dataset = []
    batch_idx=0
    
    for iterate in range(100000000): # this is very large number; but the code will through error after all saved batches are read.. 

        SAVING_PATH = f'/home/muath.abulebdeh/experiments/project13/savings/variables/preprossed_output_data/output_batch{batch_idx}.pt'
        preprocesed_outputs_batch = torch.load(SAVING_PATH, map_location=device)

        for sample in preprocesed_outputs_batch[0]: 
            preprocessed_outputs_dataset.append([])
        
        for sample_index, sample in enumerate(preprocesed_outputs_batch[0]):     
            preprocessed_outputs_dataset[sample_index + batch_size*batch_idx].append(sample)

        for sample_index, sample in enumerate(preprocesed_outputs_batch[1]):     
            preprocessed_outputs_dataset[sample_index + batch_size*batch_idx].append(sample)

       
        batch_idx += 1
        if batch_idx % 100 == 0:
            torch.save(preprocessed_outputs_dataset, '/home/muath.abulebdeh/experiments/project13/savings/variables/preprossed_output_data/preprocessed_outputs_dataset.pt')
            print(batch_idx)

        if batch_idx == 3125: 
            torch.save(preprocessed_outputs_dataset, '/home/muath.abulebdeh/experiments/project13/savings/variables/preprossed_output_data/preprocessed_outputs_dataset.pt')
            print(batch_idx)
            break 









