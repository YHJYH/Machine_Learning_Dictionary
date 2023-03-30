import os
import sys
import argparse
import torch
import torch.nn as nn
from torchvision.transforms import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import random_split, Dataset, DataLoader, SubsetRandomSampler
import torch.nn.functional as F

# Get the parent directory of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# Add the parent directory to the Python path
sys.path.append(parent_dir)

from alexnet_model import AlexNet
from other.utils import draw_loss_graph

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def data_loader(args):
    data_info = {}

    if args.dataset == 'cifar10':
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

    os.makedirs('model/%s/' % 'alexnet_model', exist_ok=True)
    model_file = 'model/%s/train_model' % 'alexnet_model'

    data = data_loader(args)

    if args.model == 'alexnet':
        Model = AlexNet(data['num_class'])
    Model.to(device)

    train_loader, test_loader = data['train_loader'], data['test_loader']

    if args.loss_type == 'CrossEntropyLoss':
        criterion = nn.CrossEntropyLoss()
    elif args.loss_type == 'NLLLoss':
        criterion = F.NLL_Loss()

    optimizer = torch.optim.SGD(Model.parameters(), lr=args.lr, momentum=args.momentum)

    best_vloss = 100
    patience = args.patience
    trigger = 0
    train_loss = []
    test_loss = []
    print("Training begin...")

    for epoch in range(args.epoch_num):
        running_loss = 0.
        last_loss = 0.
        Model.train(True)
        print('-' * 20)
        for batch_index, (img, label) in enumerate(train_loader):
            img, label = img.to(device), label.to(device)
            optimizer.zero_grad()
            pred = Model(img)
            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if batch_index % 100 == 0:
                last_loss = running_loss / 100.  # loss per batch
                print('Epoch: %d, Batch: %d. (%.0f %%)' % (
                epoch + 1, batch_index, 100. * batch_index / len(train_loader)))
                print('Train loss: %.6f' % last_loss)
                running_loss = 0.

        print('-' * 20)
        Model.train(False)
        running_vloss = 0.0
        for batch_index, (img, label) in enumerate(test_loader):
            img, label = img.to(device), label.to(device)
            pred = Model(img)
            running_vloss += criterion(pred, label).item()

        avg_vloss = running_vloss / (batch_index + 1)
        print('Train loss: %.6f, Valid loss: %.6f' % (last_loss, avg_vloss))
        train_loss.append(last_loss)
        test_loss.append(avg_vloss)
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
                break

    print('Early stopping at: %d' % (cur_epoch + 1))
    print(f'Best training loss: {last_loss}, best val loss: {best_vloss}')

    print('-' * 20)

    draw_loss_graph(train_loss=train_loss, test_loss=test_loss, loss_type=args.loss_type,
                    model_dataset=args.model + args.dataset)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-dataset', default='cifar10', type=str)  # options: cifar10
    parser.add_argument('-model', default='alexnet', type=str)  # options: alexnet

    parser.add_argument('-loss_type', default='CrossEntropyLoss', type=str)  # options: CrossEntropyLoss, NLLLoss
    parser.add_argument('-lr', default=0.001, type=float)
    parser.add_argument('-momentum', default=0.9, type=float)
    parser.add_argument('-patience', default=3, type=int)
    parser.add_argument('-epoch_num', default=50, type=int)

    args = parser.parse_args()

    train(args)



