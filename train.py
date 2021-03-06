#!/usr/bin/env python3
#-*- coding:utf-8 -*-
#coding=utf8
from __future__ import print_function, division
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import time
import argparse
from efficientnet_pytorch import EfficientNet
import evaluate
# some parameter
use_gpu = torch.cuda.is_available()

# # data_dir = '/home/user/inception_v3_retrain/'
# # batch_size = 20
# # lr = 0.01
momentum = 0.9


# num_epochs = 2
# input_size = 224
# class_num = 6
# net_name = 'efficientnet-b4'


def loaddata(pare, data_dir, batch_size, set_name, shuffle):
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize([380,380]),
            #transforms.CenterCrop(pare.input_size),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize([380,380]),
            #transforms.CenterCrop(pare.input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
         'val': transforms.Compose([
               transforms.Resize([380,380]),
               #transforms.CenterCrop(pare.input_size),
               transforms.ToTensor(),
               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in [set_name]}
    label_image = image_datasets[set_name].class_to_idx  # {"":id, }
    label_file = open(os.path.join(data_dir, "label_map.txt"),'w')
    for lab in label_image:
        content = str(lab) + " " + str(label_image[lab]) + "\n"
        label_file.write(content)
    label_file.close()
    # num_workers=0 if CPU else =1
    dataset_loaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                      batch_size=batch_size,
                                                      shuffle=shuffle, num_workers=0) for x in [set_name]}
    data_set_sizes = len(image_datasets[set_name])
    return dataset_loaders, data_set_sizes


def train_model(pare, model_ft, criterion, optimizer, lr_scheduler, num_epochs=50):
    train_loss = []
    since = time.time()
    best_model_wts = model_ft.state_dict()
    best_acc = 0.0
    model_ft.train(True)
    
    for epoch in range(num_epochs):
        dset_loaders, dset_sizes = loaddata(pare, data_dir=pare.data_dir, batch_size=pare.batch_size, set_name='train',
                                            shuffle=True)
        print('-' * 10)
        print('Data Size', dset_sizes)
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        optimizer = lr_scheduler(optimizer, epoch)

        running_loss = 0.0	
        running_corrects = 0
        count = 0
        log_file = open(os.path.join(pare.data_dir,'log.txt'),'a+')
        for data in dset_loaders['train']:
            inputs, labels = data
            # print(labels)
            # labels = torch.squeeze(labels.type(torch.LongTensor))
            # print(labels)
            # print("==============================")
            if use_gpu:
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)

            outputs = model_ft(inputs)
            # print(outputs.shape)
            # print(labels.shape)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs.data, 1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            count += 1
            if count % 30 == 0 or outputs.size()[0] < pare.batch_size:
                print('Epoch:{}: loss:{:.3f}'.format(epoch, loss.item()))
                train_loss.append(loss.item())

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / dset_sizes
        epoch_acc = running_corrects.double() / dset_sizes
        print('Epoch:{} Loss: {:.4f} Acc: {:.4f}'.format(epoch, epoch_loss, epoch_acc))
        print('Epoch:{} Loss: {:.4f} Acc: {:.4f}'.format(
            epoch, epoch_loss, epoch_acc),file=log_file)
        save_dir = pare.data_dir + '/model'
        os.makedirs(save_dir, exist_ok=True)
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = model_ft.state_dict()  
            model_ft.load_state_dict(best_model_wts)  
            model_out_path = save_dir + "/" + pare.netName + '_{}.pth'.format(epoch)
        torch.save(model_ft, model_out_path)
        # ????????????
        #if epoch % pare.evaluation_interval == 0:
         #   print("\n---- Evaluating Model ----")
         #   model = torch.load(model_out_path)
         #   print('Test Accuracy:')
         #   criterion_test = nn.CrossEntropyLoss()
         #   if use_gpu:
         #       model_ = model.cuda()
         #       criterion_test = criterion_test.cuda()
         #   evaluate.test_model(pare, model_, criterion_test)

        if epoch_acc > 0.999:
            break
        log_file.close()

    # save best model
    save_dir = pare.data_dir + '/model'
    os.makedirs(save_dir, exist_ok=True)
    model_ft.load_state_dict(best_model_wts)
    model_out_path = save_dir + "/" + pare.netName + '.pth'
    torch.save(model_ft, model_out_path)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    return train_loss, best_model_wts


# def test_model(parse, model, criterion):
#     model.eval()
#     running_loss = 0.0
#     running_corrects = 0
#     cont = 0
#     outPre = []
#     outLabel = []
#     dset_loaders, dset_sizes = loaddata(data_dir=parse.data_dir, batch_size=16, set_name='val', shuffle=False)
#     for data in dset_loaders['test']:
#         inputs, labels = data
#         labels = torch.squeeze(labels.type(torch.LongTensor))
#         inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
#         outputs = model(inputs)
#         _, preds = torch.max(outputs.data, 1)
#         loss = criterion(outputs, labels)
#         if cont == 0:
#             outPre = outputs.data.cpu()
#             outLabel = labels.data.cpu()
#         else:
#             outPre = torch.cat((outPre, outputs.data.cpu()), 0)
#             outLabel = torch.cat((outLabel, labels.data.cpu()), 0)
#         running_loss += loss.item() * inputs.size(0)
#         running_corrects += torch.sum(preds == labels.data)
#         cont += 1
#     print('Loss: {:.4f} Acc: {:.4f}'.format(running_loss / dset_sizes,
#                                             running_corrects.double() / dset_sizes))


def exp_lr_scheduler(optimizer, epoch, init_lr=0.01, lr_decay_epoch=5):
    """Decay learning rate by a f#            model_out_path ="./model/W_epoch_{}.pth".format(epoch)
#            torch.save(model_W, model_out_path) actor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.8 ** (epoch // lr_decay_epoch))
    print('LR is set to {}'.format(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./dataset", help="input train data path")
    parser.add_argument("--netName", type=str, default="efficientnet-b4", help="input net name")
    parser.add_argument("--continue_train", type=str, help="input trained model")
    parser.add_argument("--class_num", type=str, default=46, help="The class number of object")
    parser.add_argument("--input_size", type=int, default="380", help=None)
    parser.add_argument("--epochs", type=int, default=40, help="The number of model train")
    parser.add_argument("--batch_size", type=int, default=32, help="The number of data that input model")
    parser.add_argument("--lr", type=float, default=0.001, help="")
    parser.add_argument("--evaluation_interval", type=int, default=1, help="interval evaluations on validation set")
    # parser.add_argument("")
    parse_args = parser.parse_args()
    # train
    pth_map = {
        'efficientnet-b0': 'efficientnet-b0-355c32eb.pth',
        'efficientnet-b1': 'efficientnet-b1-f1951068.pth',
        'efficientnet-b2': 'efficientnet-b2-8bb594d6.pth',
        'efficientnet-b3': 'efficientnet-b3-5fb5a3c3.pth',
        'efficientnet-b4': 'efficientnet-b4.pth',
        'efficientnet-b5': 'efficientnet-b5-b6417697.pth',
        'efficientnet-b6': 'efficientnet-b6-c76e70fd.pth',
        'efficientnet-b7': 'efficientnet-b7-dcc49843.pth',
    }
    
    # model = EfficientNet.from_pretrained(parse_args.netName)
    # ?????????????????????????????????????????????????????????????????????imagenet??????????????????????????????
    if parse_args.continue_train != None:
        model = torch.load(os.path.join(parse_args.data_dir, parse_args.continue_train))
    else:
        model = EfficientNet.from_name(parse_args.netName)
        net_weight = './eff_weights/' + pth_map[parse_args.netName]
        state_dict = torch.load(net_weight)
        model.load_state_dict(state_dict)
    
    input_features = model._fc.in_features
    model._fc = nn.Linear(input_features, parse_args.class_num)
    
    criterion = nn.CrossEntropyLoss()
    if use_gpu:
        model_ft = model.cuda()
        criterion = criterion.cuda()

    optimizer = optim.SGD((model.parameters()), lr=parse_args.lr,
                          momentum=momentum, weight_decay=0.0005)
    # training
    train_loss, best_model_wts = train_model(parse_args, model, criterion, optimizer, exp_lr_scheduler, num_epochs=parse_args.epochs)
    print("training finished********")
    # test
    # print('-' * 10)
    # print('Test Accuracy:')
    # model.load_state_dict(best_model_wts)
    # criterion = nn.CrossEntropyLoss().cuda()
    # test_model(parse_args, model, criterion)
