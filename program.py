# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 15:10:03 2018

@author: iamav
"""

#general
import numpy as np
import matplotlib.pyplot as plt

#torch
import torch
import torch.backends.cudnn as cudnn
#import torch.utils.data as data
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable



#torchvision
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

#ENTER PATH HERE
TRAIN_DIR = "...\\train_set"
TEST_DIR = "...\\test_set"

def main():
    
#TRANSFORM
        transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
#TRAINING & TEST DATASETS        
        train_set = datasets.ImageFolder(root = TRAIN_DIR, transform = transform)
        test_set = datasets.ImageFolder(root = TEST_DIR, transform = transform)        
        
#LOAD DATASETS
        train_loader = DataLoader(train_set, batch_size = 4, num_workers = 4, shuffle = True)
        test_loader = DataLoader(test_set, batch_size = 4, num_workers=4, shuffle = False)
        
        
        dataloader = {'train' : train_loader, 'test': test_loader}
        
        classes = ('cat', 'dog')
        
#        print(len(train_loader))
#        print(len(dataloaders ['train']))
        
#IF GPU IS AVAILABLE
        cuda_avail = torch.cuda.is_available()
                
        model = models.resnet18(pretrained=True)     
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
        
        if cuda_avail:            
            model.cuda()
            model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
            cudnn.benchmark = True
            
        def test():
            model.eval()
            #ITERATE
            dataiter = iter(test_loader)
            images, labels = dataiter.next()
            
            test_acc = 0.0
          
            if cuda_avail:
                    images = Variable(images.cuda())
                    labels = Variable(labels.cuda())
    
            #Predict classes using images from the test set
            outputs = model(images)
            _, prediction = torch.max(outputs.data, 1)
            test_acc += torch.sum(prediction == labels.data)
            
    
    
            #Compute the average acc and loss over all test images
            test_acc = test_acc / 2023
    
            return test_acc
        
        def train(num_epochs):
            
            train_loss = 0.0
            
            for epoch in range(num_epochs):
                model.train()
                    
                dataiter = iter(train_loader)
                images, labels = dataiter.next()
        
                #Move images and labels to gpu if it's available
                if cuda_avail:
                    images = Variable(images.cuda())
                    labels = Variable(labels.cuda())
                else:
                    images, labels = Variable(images), Variable(labels)
                    
                # zero the parameter gradients
                optimizer.zero_grad()
        
                # forward,  backward and optimize
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                
                test_acc = test()
                
            # Print the metrics
            print("Epoch {},  TrainLoss: {}, TestAcc: {}".format(epoch, train_loss, test_acc))
        
        train(num_epochs =20)
        
if __name__ == '__main__':
    main()
        
