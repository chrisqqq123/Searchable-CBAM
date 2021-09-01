# train.py
#!/usr/bin/env	python3

""" train network using pytorch

author baiyu
modified by yonghye kwon
"""

import os
import sys
import argparse
from datetime import datetime
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torch.autograd import Variable
from architect import Architect
from conf import settings
from utils import get_network, get_training_dataloader, get_test_dataloader, WarmUpLR
import torch.nn.functional as F


def train(epoch, file, update_arch=True):

    net.train()
    for batch_index, (images, labels) in enumerate(cifar100_training_loader):
        if epoch <= args.warm:
            warmup_scheduler.step()

        images = Variable(images)
        labels = Variable(labels)

        labels = labels.cuda()
        images = images.cuda()

        ##### penalty #########
        penalty = True
        #######################
        if update_arch:
            if batch_index%2==1:
                loss_arch = architect.step(images, labels, penalty=penalty)
                print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
                    loss_arch.item(),
                    optimizer.param_groups[0]['lr'],
                    epoch=epoch,
                    trained_samples=batch_index * args.b + len(images),
                    total_samples=len(cifar100_training_loader.dataset)
                ))
                if batch_index%50==0:
                    file.write('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f} \n'.format(
                        loss_arch.item(),
                        optimizer.param_groups[0]['lr'],
                        epoch=epoch,
                        trained_samples=batch_index * args.b + len(images),
                        total_samples=len(cifar100_training_loader.dataset)
                    ))
            else:
                optimizer.zero_grad()
                outputs = net(images)
                loss = loss_function(outputs, labels)
                loss.backward()
                optimizer.step()
            
                print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
                    loss.item(),
                    optimizer.param_groups[0]['lr'],
                    epoch=epoch,
                    trained_samples=batch_index * args.b + len(images),
                    total_samples=len(cifar100_training_loader.dataset)
                ))
                if batch_index%50==0:
                    file.write('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f} \n'.format(
                        loss.item(),
                        optimizer.param_groups[0]['lr'],
                        epoch=epoch,
                        trained_samples=batch_index * args.b + len(images),
                        total_samples=len(cifar100_training_loader.dataset)
                    ))
            # print(net._arch_parameters)
        else:
                optimizer.zero_grad()
                outputs = net(images)
                loss = loss_function(outputs, labels)
                loss.backward()
                optimizer.step()
            
                print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
                    loss.item(),
                    optimizer.param_groups[0]['lr'],
                    epoch=epoch,
                    trained_samples=batch_index * args.b + len(images),
                    total_samples=len(cifar100_training_loader.dataset)
                ))
                if batch_index%50==0:
                    file.write('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f} \n'.format(
                        loss.item(),
                        optimizer.param_groups[0]['lr'],
                        epoch=epoch,
                        trained_samples=batch_index * args.b + len(images),
                        total_samples=len(cifar100_training_loader.dataset)
                    ))


    for name, param in net.named_parameters():
        layer, attr = os.path.splitext(name)
        attr = attr[1:]


def eval_training(epoch, file):
    net.eval()

    test_loss = 0.0 # cost function error
    correct = 0.0

    for (images, labels) in cifar100_test_loader:
        images = Variable(images)
        labels = Variable(labels)

        images = images.cuda()
        labels = labels.cuda()

        outputs = net(images)
        loss = loss_function(outputs, labels)
        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    print('Test set: Average loss: {:.4f}, Accuracy: {:.4f}'.format(
        test_loss / len(cifar100_test_loader.dataset),
        correct.float() / len(cifar100_test_loader.dataset)
    ))
    # print(len(net._arch_names[0]["alphas"]),'################################',)
    for i in range(len(net._arch_names[0]["alphas"])):
        print('Arch parameters: ',F.softmax(getattr(net, net._arch_names[0]["alphas"][i]), dim=-1) )
        print('Arch parameters: ',F.softmax(getattr(net, net._arch_names[0]["betas"][i]), dim=-1) )
    print()

    file.write('Test set: Average loss: {:.4f}, Accuracy: {:.4f} \n'.format(
        test_loss / len(cifar100_test_loader.dataset),
        correct.float() / len(cifar100_test_loader.dataset)
    ))
    for i in range(len(net._arch_names[0]["alphas"])):
        # print('Arch parameters: ',F.sigmoid(getattr(net, net._arch_names[0]["alphas"][i])))
        file.write('Arch parameters: '+str(F.softmax(getattr(net, net._arch_names[0]["alphas"][i]), dim=-1) )+'\n')
        file.write('Arch parameters: '+str(F.softmax(getattr(net, net._arch_names[0]["betas"][i]), dim=-1) )+'\n')


    return correct.float() / len(cifar100_test_loader.dataset)

def __init_weight(feature, conv_init, norm_layer, bn_eps, bn_momentum,
                  **kwargs):
    for name, m in feature.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Conv3d)):
            conv_init(m.weight, **kwargs)
            # print(m)
        elif isinstance(m, norm_layer):
            # print(m)
            m.eps = bn_eps
            m.momentum = bn_momentum
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    print('Weight initialized!')

def fix_arch(net):
    for idx, arch_name in enumerate(net._arch_names):
        for name_a, name_b in zip(arch_name['alphas'], arch_name['betas']):
            for i in range(len(getattr(net, name_a))):
                # 0 for identity, 1 for CA, 2 for SA
                a = F.softmax(getattr(net, name_a)[i])  # a size: 1*3
                # print(a.size(),a, getattr(net, name_a)[i][0])
                b = F.softmax(getattr(net, name_b)[i])
                # print(b.size(),b)
                if a[0]>0.5:
                    getattr(net, name_a)[i].data.copy_(torch.Tensor([float(5),float('-inf'),float('-inf')]).cuda())
                elif abs( a[1]-a[2] )<0.1:
                    getattr(net, name_a)[i].data.copy_(torch.Tensor([float('-inf'),float(0),float(0)]).cuda())
                else:
                    if a[1]>a[2]:
                        getattr(net, name_a)[i].data.copy_(torch.Tensor([float('-inf'),float(5),float('-inf')]).cuda())
                    else:
                        getattr(net, name_a)[i].data.copy_(torch.Tensor([float('-inf'),float('-inf'),float(5)]).cuda())
                if b[0]>0.5:
                    getattr(net, name_b)[i].data.copy_(torch.Tensor([float(5),float('-inf'),float('-inf')]).cuda())
                elif abs( b[1]-b[2] )<0.1:
                    getattr(net, name_b)[i].data.copy_(torch.Tensor([float('-inf'),float(0),float(0)]).cuda())
                else:
                    if b[1]>b[2]:
                        getattr(net, name_b)[i].data.copy_(torch.Tensor([float('-inf'),float(5),float('-inf')]).cuda())
                    else:
                        getattr(net, name_b)[i].data.copy_(torch.Tensor([float('-inf'),float('-inf'),float(5)]).cuda())
        for i in range(len(net._arch_names[0]["alphas"])):
            print('Arch parameters: ',F.softmax(getattr(net, net._arch_names[0]["alphas"][i]), dim=-1) )
            print('Arch parameters: ',F.softmax(getattr(net, net._arch_names[0]["betas"][i]), dim=-1) )


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-gpu', type=bool, default=True, help='use gpu or not')
    parser.add_argument('-w', type=int, default=2, help='number of workers for dataloader')
    parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('-s', type=bool, default=True, help='whether shuffle the dataset')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    args = parser.parse_args()

    net = get_network(args, use_gpu=args.gpu)
    architect = Architect(net)
        
    #data preprocessing:
    cifar100_training_loader = get_training_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=args.w,
        batch_size=args.b,
        shuffle=args.s
    )
    
    cifar100_test_loader = get_test_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=args.w,
        batch_size=args.b,
        shuffle=args.s
    )
    
    loss_function = nn.CrossEntropyLoss()
    parameters = []
    parameters += list(net.conv1.parameters())
    parameters += list(net.bn1.parameters())
    parameters += list(net.layer1.parameters())
    parameters += list(net.layer2.parameters())
    parameters += list(net.layer3.parameters())
    parameters += list(net.layer4.parameters())
    parameters += list(net.linear.parameters())
    # for para in parameters:
    # for name,param in net.layer1.named_parameters():
    #     print(name)
    optimizer = torch.optim.SGD(
        parameters,
        lr=args.lr,
        momentum=0.9,
        weight_decay=5e-4)
    # optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=0.2) #learning rate decay
    iter_per_epoch = len(cifar100_training_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)
    checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net+time.strftime("%Y%m%d-%H%M%S"))

    #create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    # logging #
    file = open(checkpoint_path+"/log.txt", "w") 
    file.write('model name: '+args.net+'\n')
    checkpoint_arch = os.path.join(checkpoint_path, 'arch-{epoch}-{type}.pth')
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    best_acc = 0.0
    # model search steps
    warmup = 4
    pretrain = 6
    # fixed arch parameters
    model_fixed = False

    for epoch in range(1, settings.EPOCH+pretrain+warmup):
        if epoch <= warmup:
            # warm up
            train(epoch, file, update_arch=False)
        elif epoch<=pretrain+warmup:
            # structure search
            train(epoch, file)
        else:
            if not model_fixed:
                fix_arch(net)
                model_fixed = True
                __init_weight(net, nn.init.kaiming_normal_, nn.BatchNorm2d, 1e-5, 0.1, mode='fan_in', nonlinearity='relu')
            if epoch > args.warm:
                train_scheduler.step()
            train(epoch, file, update_arch=False)
        acc = eval_training(epoch, file)

        #start to save best performance model after learning rate decay to 0.01 
        if epoch > settings.MILESTONES[1] and best_acc < acc:
            torch.save(net.state_dict(), checkpoint_path.format(net=args.net, epoch=epoch, type='best'))
            best_acc = acc
            for idx, arch_name in enumerate(net._arch_names):
                state = {}
                for name in arch_name['alphas']:
                    state[name] = getattr(net, name)
                for name in arch_name['betas']:
                    state[name] = getattr(net, name)
                torch.save(state,checkpoint_arch.format(epoch=epoch, type='best'))
            continue

        if not epoch % settings.SAVE_EPOCH:
            torch.save(net.state_dict(), checkpoint_path.format(net=args.net, epoch=epoch, type='regular'))
            if epoch<=pretrain+warmup:
                for idx, arch_name in enumerate(net._arch_names):
                    state = {}
                    for name in arch_name['alphas']:
                        state[name] = getattr(net, name)
                    for name in arch_name['betas']:
                        state[name] = getattr(net, name)
                    torch.save(state,checkpoint_arch.format(epoch=epoch, type='regular'))
    print()
    print("best_acc: ", best_acc)
    file.write("best_acc: "+ str(best_acc)+'\n')
    for i in range(len(net._arch_names[0]["alphas"])):
        # print('Arch parameters: ',F.sigmoid(getattr(net, net._arch_names[0]["alphas"][i])))
        file.write('Arch parameters: '+str(F.softmax(getattr(net, net._arch_names[0]["alphas"][i]),dim=-1) )+'\n')
        file.write('Arch parameters: '+str(F.softmax(getattr(net, net._arch_names[0]["betas"][i]),dim=-1) )+'\n')
    file.close()