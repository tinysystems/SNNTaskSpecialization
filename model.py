import torch
import torch.nn as nn
from layers import *
import torch.nn.functional as F

from setting import (
    STEPS
)




steps = STEPS




class MyActivation(nn.Module):
    def __init__(self):
        super().__init__()
        #self.c = nn.Parameter(torch.tensor(10))
        # self.k = nn.Parameter(torch.tensor(0.2))
        self.k = 0.2
        
    def forward(self, x):
        return torch.sigmoid(1*(x - self.k)) # You may reduce the slope here as the accuracy was still low for some subSNNs



class sub_VGG9(nn.Module):  # Example net for CIFAR10
    def __init__(self, scale_factor=4):
        super(sub_VGG9, self).__init__()

        self.scale_factor = scale_factor
                
        self.layer0 = nn.Conv2d(3, int(128/self.scale_factor), 3, 1, 1, bias=None)
        self.bn0 = tdBatchNorm(int(128/self.scale_factor))
        self.layer1 = nn.Conv2d(int(128/self.scale_factor), int(256/self.scale_factor), 3, 1, 1, bias=None)
        self.bn1 = tdBatchNorm(int(256/self.scale_factor))
        self.pool1 = nn.AvgPool2d(2)
        
        self.layer2 = nn.Conv2d(int(256/self.scale_factor), int(512/self.scale_factor), 3, 1, 1, bias=None)
        self.bn2 = tdBatchNorm(int(512/self.scale_factor))
        self.layer3 = nn.Conv2d(int(512/self.scale_factor), int(512/self.scale_factor), 3, 1, 1, bias=None)
        self.bn3 = tdBatchNorm(int(512/self.scale_factor))
        self.pool2 = nn.AvgPool2d(2)
        
        self.layer4 = nn.Conv2d(int(512/self.scale_factor), int(1024/self.scale_factor), 3, 1, 1, bias=None)
        self.bn4 = tdBatchNorm(int(1024/self.scale_factor))
        self.layer5 = nn.Conv2d(int(1024/self.scale_factor), int(512/self.scale_factor), 3, 1, 1, bias=None)
        self.bn5 = tdBatchNorm(int(512/self.scale_factor))
        
        self.layer6 = nn.Linear(8 * 8 * int(512/self.scale_factor), int(1024/self.scale_factor))
        self.layer7 = nn.Linear(int(1024/self.scale_factor), int(512/self.scale_factor))
        self.layer8 = nn.Linear(int(512/self.scale_factor), int(10/10))

        self.layer0_s = tdLayer(self.layer0, self.bn0)
        self.layer1_s = tdLayer(self.layer1, self.bn1)
        self.pool1_s = tdLayer(self.pool1)

        self.layer2_s = tdLayer(self.layer2, self.bn2)
        self.layer3_s = tdLayer(self.layer3, self.bn3)
        self.pool2_s = tdLayer(self.pool2)
        
        self.layer4_s = tdLayer(self.layer4, self.bn4)
        self.layer5_s = tdLayer(self.layer5, self.bn5)
        
        self.layer6_s = tdLayer(self.layer6)
        self.layer7_s = tdLayer(self.layer7)
        self.layer8_s = tdLayer(self.layer8)

        self.spike = LIFSpike(membrane_out=True)
        self.Activation = MyActivation()
        

        self.activity = []
   
    def forward(self, x):

        self.activity = []
        num_output_feature_maps = []
        neurons_per_layer = []

        
        x = self.layer0_s(x)
        x, _ = self.spike(x)

        self.activity.append(x.detach().mean(4).mean(0).sum()) # first take the mean from all steps, then take the mean from all batches then sum the results (of all elements in all channels)
        neurons_per_layer.append(x.shape[1]*x.shape[2]*x.shape[3]) # channels*elements_dim1*elements_dim2
        num_output_feature_maps.append(x.shape[1])

        # print(self.activity[0].shape)
        x = self.layer1_s(x)
        x, _ = self.spike(x)
        self.activity.append(x.detach().mean(4).mean(0).sum())
        neurons_per_layer.append(x.shape[1]*x.shape[2]*x.shape[3])
        num_output_feature_maps.append(x.shape[1])
        # print(self.activity[1].shape)

        x = self.pool1_s(x)
        x, _ = self.spike(x)

        x = self.layer2_s(x)
        x, _ = self.spike(x)
        self.activity.append(x.detach().mean(4).mean(0).sum())
        neurons_per_layer.append(x.shape[1]*x.shape[2]*x.shape[3])
        num_output_feature_maps.append(x.shape[1])
        # print(self.activity[2].shape)

        x = self.layer3_s(x)
        x, _ = self.spike(x)
        self.activity.append(x.detach().mean(4).mean(0).sum())
        neurons_per_layer.append(x.shape[1]*x.shape[2]*x.shape[3])
        num_output_feature_maps.append(x.shape[1])
        # print(self.activity[3].shape)

        x = self.pool2_s(x)
        x, _ = self.spike(x)

        x = self.layer4_s(x)
        x, _ = self.spike(x)
        self.activity.append(x.detach().mean(4).mean(0).sum())
        neurons_per_layer.append(x.shape[1]*x.shape[2]*x.shape[3])
        num_output_feature_maps.append(x.shape[1])
        # print(self.activity[4].shape)

        x = self.layer5_s(x)
        x, _ = self.spike(x)
        self.activity.append(x.detach().mean(4).mean(0).sum())
        neurons_per_layer.append(x.shape[1]*x.shape[2]*x.shape[3])
        num_output_feature_maps.append(x.shape[1])
        # print(self.activity[5].shape)
 
        x = x.view(x.shape[0], -1, x.shape[4])
        x = self.layer6_s(x)
        x, _ = self.spike(x)
        self.activity.append(x.detach().mean(2).mean(0).sum())
        neurons_per_layer.append(x.shape[1]) # in FC, we have only one dimension
        num_output_feature_maps.append(x.shape[1])
        # print(self.activity[6].shape)

        x = self.layer7_s(x)
        x, _ = self.spike(x)
        self.activity.append(x.detach().mean(2).mean(0).sum())
        neurons_per_layer.append(x.shape[1])
        num_output_feature_maps.append(x.shape[1])
        # print(self.activity[7].shape)

        x = self.layer8_s(x)
        x, u = self.spike(x)
        self.activity.append(x.detach().mean(2).mean(0).sum())
        neurons_per_layer.append(x.shape[1])
        num_output_feature_maps.append(x.shape[1])
        # print(self.activity[8].shape)

        
        return x, u, self.activity, neurons_per_layer, num_output_feature_maps






class weighted_average_voting_1(nn.Module):  # This is for implementing the aggregation function
    def __init__(self):
        super(weighted_average_voting_2, self).__init__()
        self.layer1 = nn.Linear(10, 10)        

        self.dropout1 = nn.Dropout(p=0.2)
        
        # self.activation = nn.ReLU()
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.layer1(x)

        return x


class weighted_average_voting_2(nn.Module):  # This is for implementing the aggregation function
    def __init__(self):
        super(weighted_average_voting_2, self).__init__()
        self.layer1 = nn.Linear(10, 1000)        
        self.layer2 = nn.Linear(1000, 10)

        self.dropout1 = nn.Dropout(p=0.2)
        
        # self.activation = nn.ReLU()
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        # x = self.dropout1(x)

        x = self.layer2(x)        
        return x




class weighted_average_voting_3(nn.Module):  # This is for implementing the aggregation function
    def __init__(self):
        super(weighted_average_voting_3, self).__init__()
        self.layer1 = nn.Linear(13, 1000)        
        self.layer2 = nn.Linear(1000, 13)

        self.dropout1 = nn.Dropout(p=0.2)
        
        # self.activation = nn.ReLU()
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        # x = self.dropout1(x)

        x = self.layer2(x)        
        return x

