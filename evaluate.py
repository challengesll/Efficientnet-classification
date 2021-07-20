from __future__ import print_function, division
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from train import *
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import time
#import os

use_gpu = torch.cuda.is_available()



def test_model(parse, model, criterion):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    cont = 0
    outPre = []
    outLabel = []
    dset_loaders, dset_sizes = loaddata(parse, data_dir=parse.data_dir, batch_size=1, set_name='test', shuffle=False)
    for data in dset_loaders['test']:
        inputs, labels = data
        #print(labels)
        #labels = torch.squeeze(labels.type(torch.LongTensor))
        labels = torch.LongTensor(labels)
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)
        #print(preds)
        #print("label:",labels)
        #print("预测结果为：", preds.data)
        #print("真值结果为：", labels.data)
        loss = criterion(outputs, labels)
        if cont == 0:
            outPre = outputs.data.cpu()
            outLabel = labels.data.cpu()
        else:
            outPre = torch.cat((outPre, outputs.data.cpu()), 0)
            outLabel = torch.cat((outLabel, labels.data.cpu()), 0)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        cont += 1
    print('Loss: {:.4f} Acc: {:.4f}'.format(running_loss / dset_sizes,
                                            running_corrects.double() / dset_sizes))

if __name__ == '__main__':
    # 参数定义
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="dataset", help="input train data path")
    parser.add_argument("--netName", type=str, default="efficientnet-b4", help="input net name")
    parser.add_argument("--class_num", type=str, default=46, help="The class number of object")
    parser.add_argument("--input_size",type=int, default="380", help=None)
    parser.add_argument("--epochs", type=int, default=2, help="The number of model train")
    parser.add_argument("--batch_size", type=int, default=1, help="The number of data that input model")
    parser.add_argument("--lr", type=float, default=0.01, help="")
    parser.add_argument("--model_dir", type=str, default="dataset/model/efficientnet-b4.pth")
    parser_arg = parser.parse_args()
    pth_map = {
        'efficientnet-b0': 'efficientnet-b0-355c32eb.pth',
        'efficientnet-b1': 'efficientnet-b1-f1951068.pth',
        'efficientnet-b2': 'efficientnet-b2-8bb594d6.pth',
        'efficientnet-b3': 'efficientnet-b3-5fb5a3c3.pth',
        'efficientnet-b4': 'efficientnet-b4-6ed6700e.pth',
        'efficientnet-b5': 'efficientnet-b5-b6417697.pth',
        'efficientnet-b6': 'efficientnet-b6-c76e70fd.pth',
        'efficientnet-b7': 'efficientnet-b7-dcc49843.pth',
    }
    model = torch.load(parser_arg.model_dir)
    # test
    criterion = nn.CrossEntropyLoss()
    if use_gpu:
        model = model.cuda()
        criterion = criterion.cuda()
    print('-' * 10)
    print('Test Accuracy:')
    # model_ft.load_state_dict(best_model_wts)
    # criterion = nn.CrossEntropyLoss().cuda()
    test_model(parser_arg, model, criterion)
