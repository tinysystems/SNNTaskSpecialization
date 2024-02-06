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

import sys


from tensorboardX import SummaryWriter

from setting import (
    STEPS
)



train_loss_hist = []
test_loss_hist = []
test_acc_hist = []


def train(args, model, device, train_loader, optimizer, epoch, writer, divide_loss_by_20=False):


    loss_fn = nn.BCEWithLogitsLoss().to(device) # BCELoss expects an output between 0 and 1

    model.train()

    steps = STEPS

    # I am training using gradient accumulation here (if needed) to be able to fit the data in the gpu memory; 

    num_accumulation_step =  int(64/(args.batch_size)) 


    for batch_idx, (data, target) in enumerate(train_loader):

        data, target = data.to(device), target.to(device)

        target = torch.where(target==subSNN_index, 1, 0).to(dtype=torch.float32)

        # necessary for general dataset: broadcast input
        data, _ = torch.broadcast_tensors(data, torch.zeros((steps,) + data.shape)) 

        data = data.permute(1, 2, 3, 4, 0)
        
        data = data + (torch.rand(data.shape).to(device) / 10) 

        output, mem, _, _, _ = model(data)

        loss = 0
        for n in range(steps):
            loss+= loss_fn(mem[:,0,n], target)

 
        loss = loss/num_accumulation_step
        if divide_loss_by_20 == True:
            loss = loss/20 

        loss.backward()
               
        # Update Optimizer
        if ((batch_idx + 1) % num_accumulation_step == 0) or (batch_idx + 1 == len(train_loader)):
            optimizer.step()
            optimizer.zero_grad()



        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data / steps), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item())) 
            print(f'output.sum(): {output.sum()}')

            
            writer.add_scalar('Train Loss /batchidx', loss, batch_idx + len(train_loader) * epoch)
   
    train_loss_hist.append(loss.item())        
    torch.save(train_loss_hist, MODEL_PATH+"train_loss_hist.pt")

def test(args, model, device, test_loader, epoch, writer):

    loss_fn = nn.BCEWithLogitsLoss().to(device)
    # loss_fn = nn.BCELoss().to(device)

    layers_avg_activity = [] 
    for i in range(9): # 9 layers
        layers_avg_activity.append(0)


    model.eval()
    test_loss = 0
    correct = 0
    isEval = False
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            target = torch.where(target==subSNN_index, 1, 0).to(dtype=torch.float32)

            data, _ = torch.broadcast_tensors(data, torch.zeros((steps,) + data.shape))
            data = data.permute(1, 2, 3, 4, 0)

            output, mem, activity, _, _ = model(data)
    
            for n in range(steps):
                test_loss += loss_fn(mem[:,0,n], target).item()

            pred = torch.round(output.mean(dim=2)) 


            correct += pred.eq(target.view_as(pred)).sum().item()

            for i, layer_activity in enumerate(activity):
                layers_avg_activity[i] = layers_avg_activity[i] + layer_activity

    test_loss /= len(test_loader.dataset)

    writer.add_scalar('Test Loss /epoch', test_loss, epoch)
    writer.add_scalar('Test Acc /epoch', 100. * correct / len(test_loader.dataset), epoch)
    for i, (name, param) in enumerate(model.named_parameters()):
        if '_s' in name:
            writer.add_histogram(name, param, epoch)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    test_loss_hist.append(test_loss)
    test_acc_hist.append(100. * correct / len(test_loader.dataset))
    torch.save(test_loss_hist, MODEL_PATH+"test_loss_hist.pt")
    torch.save(test_acc_hist, MODEL_PATH+"test_acc_hist.pt")

    test_accuracy = 100. * correct / len(test_loader.dataset)

    return test_accuracy, layers_avg_activity

def baseline_test(model, device, test_loader):

    model.eval()
    test_loss = 0
    correct = 0
    isEval = False
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            target = torch.where(target==subSNN_index, 1, 0).to(dtype=torch.float32)

            data, _ = torch.broadcast_tensors(data, torch.zeros((steps,) + data.shape))
            data = data.permute(1, 2, 3, 4, 0)

            output, _, _ = model(data)

            #pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            output_threshold = 0.2
            pred = torch.where(output[:, subSNN_index]>output_threshold, 1, 0).to(dtype=torch.float32)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_accuracy = 100. * correct / len(test_loader.dataset)

    return test_accuracy

def adjust_learning_rate(args, optimizer, epoch): # Now the learning rate is not adjusted; you may need to add this to your code later..
    """Sets the learning rate to the initial LR decayed by 10 every 20 epochs"""
    lr = args.lr * (0.1 ** (epoch // 35))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    torch.save(lr, MODEL_PATH+"lr.pt")

def main():
    # Training settings

    parser = argparse.ArgumentParser(description='PyTorch cifar10-vgg9 Example - subSNNS')

    parser.add_argument('--subSNN-index', type=int, default=0, metavar='N',
                        help='subSNN index (default: 0)') 

    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)') # batch size here is the downloaded; the computed batch size is fized to 64
    parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=250, metavar='N',
                        help='number of epochs to train (default: 10)') # we run the baseline for approximately 170 epochs..
    parser.add_argument('--lr', type=float, default=5e-5, metavar='LR',
                        help='learning rate (default: 0.01)') # changed from 1e-4 to 5e-5 similar to the codes I had un jupyter
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    #### Edited 
    parser.add_argument('--reset-model', type=int, default=1)
    parser.add_argument('--scale-factor', type=int, default=16)
    ## I removoed from this code some conditions that are not needed to speed up the code (e..g., activation_enbled, summation_enabled, etc)
    #### 

    args = parser.parse_args()



    use_cuda = not args.no_cuda and torch.cuda.is_available()

    print(f'CUDA used: {use_cuda}')

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 2, 'pin_memory': True} if use_cuda else {}


    TENSORBOARD_LOG_PATH = '/home/muath.abulebdeh/experiments/project13/python_scripts/subSNN_tdBN_training/STBP-simple/execution_on_hpc/cifar10_vgg9/TENSORBOARD_LOG' 
    writer = SummaryWriter(TENSORBOARD_LOG_PATH)


    # Details for sub-SNNs and sub-Tasks

    global num_classes 
    global subSNN_index
    global num_layers 

    global MODEL_PATH 
    global BASELINE_MODEL_PATH 

    global global_epoch



    num_classes = 10
    subSNN_index = args.subSNN_index
    num_layers = 9

    print(f'subSNN index: {subSNN_index}')

    model_index = subSNN_index
    MODEL_PATH = f'/home/muath.abulebdeh/experiments/project13/savings/variables/cifar10_vgg9_subSNN_{model_index}'
    BASELINE_MODEL_PATH = "/home/muath.abulebdeh/experiments/project13/savings/variables/cifar10_vgg9_"

    

    # load the test and train datasets


    # Create a train set
    DATA_PATH = '/home/muath.abulebdeh/colab_projects/data/cifar10/'

    train_dataset_cifar10 = datasets.CIFAR10(DATA_PATH, 
                            train=True, 
                            download=True,
                            transform=transforms.Compose([
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomCrop(32, padding=4),
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
                         )

    train_loader = torch.utils.data.DataLoader(train_dataset_cifar10, batch_size=args.batch_size, shuffle=True, **kwargs) 

    # Give a weight for every sample in the dataset
    target_class = subSNN_index
    samples_weights = []
    for image, label in train_dataset_cifar10:
        if label==target_class: 
            samples_weights.append(0.5)
        else: 
            samples_weights.append(0.5/9)
    # Create the sampler with the same length of the dataset (everysample will have a weight - they do not need necessarily to be equal to 1, pytorch does this internally)
    sampler = torch.utils.data.WeightedRandomSampler(weights=samples_weights, num_samples=len(train_dataset_cifar10), replacement=True)
    # Create biases samples
    train_loader_50_50 = torch.utils.data.DataLoader(train_dataset_cifar10, batch_size=args.batch_size,drop_last=True, sampler=sampler) 



    # Test set 50% for target class and 50% for the alien class (all other classes)
    test_dataset_cifar10 = datasets.CIFAR10(DATA_PATH, train=False, download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
                        ]))

    # Give a weight for every sample in the dataset
    target_class = subSNN_index
    samples_weights = []
    for image, label in test_dataset_cifar10:
        if label==target_class: 
            samples_weights.append(0.5)
        else: 
            samples_weights.append(0.5/9)

    sampler = torch.utils.data.WeightedRandomSampler(weights=samples_weights, num_samples=len(test_dataset_cifar10), replacement=True)
    # Create biases samples
    biasely_distributed_test_loader = torch.utils.data.DataLoader(test_dataset_cifar10, batch_size=args.batch_size,drop_last=True, sampler=sampler) 

    # Creare the test set; the test loader is balanced here
    equally_distributed_test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(DATA_PATH, train=False, 
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
                        ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)


    


    # Measure the accuracy of the baseline model for this sub-task; baseline accuracy 
    # model = VGG9().to(device)
    # model.load_state_dict(torch.load(BASELINE_MODEL_PATH+'model.pt'))
    baseline_test_accuracy = 95 #baseline_test(model, device, biasely_distributed_test_loader)
    print(f'Baseline test accuracy: {baseline_test_accuracy} => Change this later..')
    model = 0


    
    S = [args.scale_factor] # I make this flexible to use the code in a better way..



    #### Edited 
    if args.reset_model == 0:

        scale_factor_index = torch.load(MODEL_PATH+"scale_factor_index.pt")
        scale_factor = S[scale_factor_index]
        model = sub_VGG9(scale_factor=scale_factor).to(device)      
        print(f'Scaling factor: {scale_factor}') 
        model.load_state_dict(torch.load(MODEL_PATH+'model.pt'))
        global_epoch = torch.load(MODEL_PATH+'global_epoch.pt')
        train_loss_hist = torch.load(MODEL_PATH+"train_loss_hist.pt")
        test_loss_hist = torch.load(MODEL_PATH+"test_loss_hist.pt")
        test_acc_hist = torch.load(MODEL_PATH+"test_acc_hist.pt")
        lr = torch.load(MODEL_PATH+"lr.pt")

    else: 

        training_completed_flag = 0
        torch.save(training_completed_flag, MODEL_PATH+'training_completed_flag.pt')

        global_epoch = 0
        torch.save(global_epoch, MODEL_PATH+'global_epoch.pt')
        scale_factor_index = 0
        torch.save(scale_factor_index, MODEL_PATH+'scale_factor_index.pt')
        scale_factor = S[scale_factor_index] # Change this later to 1
        model = sub_VGG9(scale_factor=scale_factor).to(device)
        print(f'Scaling factor: {scale_factor}') 



        torch.save(args.lr, MODEL_PATH+"lr.pt")
    

    first_epoch = global_epoch 

    SET3_EPOCH = args.epochs # make the batching homogenous only if we reach a specifc accuracy...
    stop_hetero_batching = 0



    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)   
    max_accuracy = 0     

    for epoch in range(first_epoch, args.epochs + 1):

        if (epoch == SET3_EPOCH) or (stop_hetero_batching==1): # this equals to 40 not >40 to create this data loader only once (not at every epoch greater than 40) 
            train_loader = 0 # clean it from memory
            SET3_EPOCH = epoch-3 # this is to not enter this condition in future epochs
            

        if (epoch<SET3_EPOCH) & (stop_hetero_batching==0): # later, you may make SET3_EPOCH to a higher value; note that stop_hetero_batching is 1 when acc_50_50>90
            # Train the model for 1 epoch
            if epoch%2 == 0: 
                train(args=args, model=model, device=device, train_loader=train_loader_50_50, optimizer=optimizer, epoch=epoch, writer=writer)
            else: 
                train(args=args, model=model, device=device, train_loader=train_loader, optimizer=optimizer, epoch=epoch, writer=writer, divide_loss_by_20=True)                   

        else: # after epoch SET3_EPOCH 
            train(args=args, model=model, device=device, train_loader=train_loader_50_50, optimizer=optimizer, epoch=epoch, writer=writer)
            
        # Test the model after narrowing the network;
        print('Testing with equally distrubited batches:')         
        _, _ = test(args, model, device, equally_distributed_test_loader, epoch, writer)
        print('Testing with biasely distrubited batches:')
        test_accuracy, _ = test(args, model, device, biasely_distributed_test_loader, epoch, writer)



        

        
        MODELS_HISTORY_PATH = f'/home/muath.abulebdeh/experiments/project13/savings/variables/cifar10_vgg9_scale_factor{scale_factor}/cifar10_vgg9_subSNN_{model_index}_'
        if test_accuracy > 90:
            stop_hetero_batching = 1
            if test_accuracy > 100:
                torch.save(model.state_dict(), MODELS_HISTORY_PATH+f'acc_{test_accuracy}_scale_factor{scale_factor}_epoch{global_epoch}.pt')
                sys.exit()
            else: 
                torch.save(model.state_dict(), MODELS_HISTORY_PATH+f'acc_{test_accuracy}_scale_factor{scale_factor}_epoch{global_epoch}.pt')
            
        # Here I want to save the model state regardless of the accuracy (save the maximum accuracy achieved).. 
        if (test_accuracy>max_accuracy) & (args.save_model==True): 
            torch.save(model.state_dict(), MODELS_HISTORY_PATH+f'acc_{test_accuracy}_scale_factor{scale_factor}_epoch{global_epoch}.pt')
            max_accuracy = test_accuracy


        global_epoch+=1

        

        # Saving the pruned model
        if epoch % 2 == 0: 
            if (args.save_model):
                torch.save(model.state_dict(), MODEL_PATH+'model.pt')
                torch.save(global_epoch, MODEL_PATH+'global_epoch.pt')
            torch.save(test_accuracy, MODEL_PATH+f'test_accuracy_of_sf{S[scale_factor_index]}')
    
    if test_accuracy > baseline_test_accuracy: 
        print(f'Training is complete; accuracy = {test_accuracy} which is greater than {baseline_test_accuracy}')
        print(f'Scale factor: {scale_factor}')
        print('...')
        training_completed_flag = 1
        torch.save(scale_factor_index, MODEL_PATH+"scale_factor_index.pt")
        torch.save(model.state_dict(), MODEL_PATH+f'model_scale_factor_{scale_factor}.pt')
        torch.save(training_completed_flag, MODEL_PATH+'training_completed_flag.pt')
        training_successful = 1
        torch.save(training_successful, MODEL_PATH+f'training_successful_scale_factor{scale_factor}.pt')
        # test_acc_hist = torch.load(MODEL_PATH+f'test_acc_hist_scale_factor{scale_factor}.pt')
        # scale_factor_index+=1

    else: 

        print(f'Training is complete; accuracy = {test_accuracy} which is less than {baseline_test_accuracy}')
        print(f'Scale factor: {scale_factor}')
        print('...')
        training_completed_flag = 1
        torch.save(scale_factor_index, MODEL_PATH+"scale_factor_index.pt")
        torch.save(model.state_dict(), MODEL_PATH+f'model_scale_factor_{scale_factor}.pt')
        torch.save(training_completed_flag, MODEL_PATH+'training_completed_flag.pt')
        training_successful = 0
        torch.save(training_successful, MODEL_PATH+f'training_successful_scale_factor{scale_factor}.pt')


            

    writer.close()

if __name__ == '__main__':
    main()

