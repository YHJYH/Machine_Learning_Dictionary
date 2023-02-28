import os
import argparse
import torch
import torch.nn as nn
from torchvision.transforms import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import random_split, Dataset, DataLoader, SubsetRandomSampler

from dataset import Caltech256Dataset
from model import EfficientNetB0, EfficientNetCustomize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_path_dir= os.getcwd()

def data_loader(args):
    data_info = {}
    
    if args.dataset == 'caltech256':
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        dataset = Caltech256Dataset(root=data_path_dir+'\\caltech256\\256_ObjectCategories', download=True, transform=transform)
        
        dataset_size = len(dataset)
        train_size = int(0.8 * dataset_size)
        test_size = dataset_size - train_size
        # train_indices = list(range(train_size))
        # test_indices = list(range(train_size, dataset_size))
        
        train_set, test_set = random_split(dataset, [train_size, test_size])
        
        # train_sampler = SubsetRandomSampler(train_indices)
        # test_sampler = SubsetRandomSampler(test_indices)
        
        train_loader = DataLoader(dataset=train_set, batch_size=1, shuffle=True)
        test_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=True)
        
        num_class = 256
        height, width, channel = 224, 224, 3
    
    elif args.dataset == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        dataset = CIFAR10('/files/', train=True, download=True, transform=transform)
        
        train_set, test_set = random_split(dataset, [40000, 10000])
        
        train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=64, shuffle=True)
        
        num_class = 10
        height, width, channel = 32, 32, 3
    
    data_info['train_loader'] = train_loader
    data_info['test_loader'] = test_loader
    data_info['num_class'] = num_class
    data_info['img_size'] = (channel, height, width)
        
    return data_info


def train(args):
    
    print(args)
    
    os.makedirs('model/%s/' % 'efficientnet_model', exist_ok=True)
    model_file = 'model/%s/train_model' % 'efficientnet_model'
    
    data = data_loader(args)
    
    if args.model == 'b0':
        Model = EfficientNetB0(data['num_class'])
    elif args.model == 'customize':
        Model = EfficientNetCustomize(data['num_class'])
    Model.to(device)
    
    train_loader, test_loader = data['train_loader'], data['test_loader']
    
    criterion = nn.CrossEntropyLoss()
    
    optimizer = torch.optim.SGD(Model.parameters(), lr=args.lr, momentum=args.momentum)
    
    best_vloss = 100
    patience = args.patience
    trigger = 0
    print("Training begin...")
    for epoch in range(args.epoch_num):
        running_loss = 0.
        last_loss = 0.
        Model.train(True)
        print('-'*20)
        for batch_index, (img, label) in enumerate(train_loader):
            img, label = img.to(device), label.to(device)
            optimizer.zero_grad()
            # print(len(Model(img)), type(Model(img)), Model(img)[0], Model(img)[1])
            pred = Model(img)
            # pred_label =torch.argmax(pred)
            # print(pred)
            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if batch_index % 100 == 0:
                last_loss = running_loss / 100. # loss per batch
                print('Epoch: %d, Batch: %d. (%.0f %%)' % (epoch+1, batch_index, 100.*batch_index/len(train_loader)))
                print('Train loss: %.6f' % last_loss)
                torch.save({
                    'MODEL': Model.state_dict(),
                    'optimizer': optimizer.state_dict()
                    }, model_file)
                running_loss = 0.
                               
        print('-'*20)
        Model.train(False)
        running_vloss = 0.0
        for batch_index, (img, label) in enumerate(test_loader):
            
            img, label = img.to(device), label.to(device)
            
            pred = Model(img)[1]
            running_vloss += criterion(pred, label).item()
        
        avg_vloss = running_vloss / (batch_index+1)
        print('Train loss: %.6f, Valid loss: %.6f' % (last_loss, avg_vloss))
        if avg_vloss <= best_vloss:
            trigger = 0
            print('Updating model file...')
            best_vloss = avg_vloss
            torch.save({
                'MODEL': Model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'dev_loss': best_vloss
                }, model_file)
            cur_epoch = epoch
        else:
            trigger += 1
            if trigger >= patience:
                # print('Early stopping at: %d' % (cur_epoch+1))    
                break
        
        print('Early stopping at: %d' % (cur_epoch+1)) 
        print(f'Current training loss: {last_loss}, current val loss: {best_vloss}')
                
    print('-'*20)
        
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-dataset', default='cifar10', type=str) # options: cifar10, caltech256
    parser.add_argument('-model', default='customize', type=str) # options: customize, b0
    
    parser.add_argument('-lr', default=0.001, type=float)
    parser.add_argument('-momentum', default=0.9, type=float)
    parser.add_argument('-patience', default=3, type=int)
    parser.add_argument('-epoch_num', default=20, type=int)
    
    args = parser.parse_args()
    
    train(args)
    