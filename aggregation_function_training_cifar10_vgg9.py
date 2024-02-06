from __future__ import print_function
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

from model import *
from dataset import NMNIST

import sys




"""

This is a second stage training for the parameters of the average weighted voting used to aggregate the results of all sub-tasks..

"""



device = 'cuda' if torch.cuda.is_available() else 'cpu'


## dataLoader
def dataLoader_custom(batch_size = 64):

    # Dataloader arguments
    
    dtype = torch.float
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

  
    # Donwload the dataset 
    DATA_PATH = '/home/muath.abulebdeh/experiments/project13/savings/variables/preprossed_output_data/preprocessed_outputs_dataset.pt'
    dataset = torch.load(DATA_PATH, map_location=device)
        
    num_samples = len(dataset)
    
    
    num_samples_train = int(90*num_samples/100) # 90%
    
    train_set = []
    test_set = []
    
    counter=0
    for sample in dataset:    
        if counter < num_samples_train: 
            train_set.append(sample)
        else: 
            test_set.append(sample)        
        counter+=1

    train_loader = DataLoader(train_set, 
                              batch_size=batch_size, 
                              drop_last=True,
                              shuffle=True ) # shuffle is mutual exclsive with sampler


    test_loader = DataLoader(test_set, 
                            batch_size=batch_size, 
                            drop_last=True,
                            shuffle=True ) # shuffle is mutual exclsive with sampler


    dtype = torch.float
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
 
    return train_loader, test_loader, batch_size, dtype, device


batch_size = 64
train_loader, test_loader, batch_size, dtype, device = dataLoader_custom(batch_size=batch_size)







def measure_accuracy(intermediate_output, t): 


        output = aggregation_function(intermediate_output)

        accuracy = (output.argmax(dim=1) == t).sum().to(torch.float) * 100 / batch_size

        return accuracy





def measure_accuracy_argmax_aggregator(intermediate_output, t): 


        output = intermediate_output

        accuracy = (output.argmax(dim=1) == t).sum().to(torch.float) * 100 / batch_size

        return accuracy









testing_accuracy = 0
num_samples = 0

## Create and train the aggregator function

aggregation_function = weighted_average_voting_2().to(device)
AGGR_FN_PATH = '/home/muath.abulebdeh/experiments/project13/savings/variables/aggtrgation_fn.pt'
reset = 1
if reset == 0: 
    aggregation_function.load_state_dict(torch.load(AGGR_FN_PATH, map_location=device))


def one_hot_conversion(a): 
    a = np.array(a.cpu().numpy())
    #b = np.zeros((a.size, a.max() + 1))
    b = np.zeros((a.size, 9 + 1))
    b[np.arange(a.size), a] = 1
    return torch.tensor(b).to(torch.float32).to(device)


loss_fn = nn.MSELoss().to(device) 
optimizer = optim.Adam(aggregation_function.parameters(), lr=1e-5)        

for epoch in range(400):
    
    aggregation_function.train()
    
    for batch_idx, (data, target) in enumerate(train_loader):

        intermediate_output = data

        output = aggregation_function(intermediate_output)

        loss = F.cross_entropy(output, target) # Maybe it is more accuracte to use mse to be consistent with the paper.. 
        # target_converted = one_hot_conversion(target).to(device)
        # loss = loss_fn(output, target_converted) # Here 0 index is always used for all subSNNs

        loss.backward()               
        optimizer.step()
        optimizer.zero_grad()


    num_samples = 0
    testing_accuracy = 0
    for test_data, test_target in test_loader: 
        test_data, test_target = test_data.to(device), test_target.to(device)
        testing_accuracy += measure_accuracy(test_data, test_target)
        num_samples += 1



    #testing_accuracy /= batch_size
    training_accuracy = measure_accuracy(data, target)

    
    if testing_accuracy/num_samples > 87: 
        torch.save(aggregation_function.state_dict(), f'/home/muath.abulebdeh/experiments/project13/savings/variables/aggtrgation_fn_acc{testing_accuracy/num_samples}_epoch{epoch}.pt') 
    
    print('')
    print(f'Epoch:{epoch}, iteration: {batch_idx}')
    print(f'traning loss: {loss.item()}')
    print(f'avg test accuracy (all test data): {testing_accuracy/num_samples}')
    print(f'training accuracy: {training_accuracy}')
    print('')




 


# ## Save the trained parameters of the aggregation function