import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#from train import *
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import time
from PIL import Image
import json
use_gpu = torch.cuda.is_available()


def inference(parse, model, criterion):
    model.eval()
    # running_loss = 0.0
    # running_corrects = 0
    cont = 0
    outPre = []
    # outLabel = []
    print("loading inference data")
    transformer = transforms.Compose([
            transforms.Resize([380,380]),
            #transforms.CenterCrop(parse.input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    # 加载图像
    category = torch.nn.Softmax(dim=1)
    label_map = {}
    label_file = open("./dataset/label_map.txt", 'r',encoding='utf8')
    for line in label_file.readlines():  # cat 0
        label, id = line.strip().split(" ")[0], int(line.strip().split(" ")[1])
        label_map.update({id:label})

    # for instance in instances:
    #     image_id = instance["id"]
    #     key_frame = instance["key_frame"]
    #     print("实例：", image_id)
    #     all_image = torch.zeros([1, 3])
    #     for img in os.listdir(os.path.join(parse.image_dir, image_id)):
    image_path = os.path.join(parse.image_dir)
    image = Image.open(image_path)
    image_tensor = transformer(image).float()
    image_tensor = image_tensor.unsqueeze(0)

    inputs = Variable(image_tensor).cuda()
    outputs = model(inputs).cuda()
    print(outputs.data)
    #all_image += outputs.data
            # print("all_image:",all_image)
        # outputs = category(all_image)

    _, preds = torch.max(outputs, 1)
        # 模型输出
    outLabel = label_map[int(preds)]
    #   输出结果写入json文件
    return outLabel



if __name__ == '__main__':
    # some parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default='dataset/infer.jpeg', help='inference the category of image')
    parser.add_argument("--netName", type=str, default="efficientnet-b4", help="input net name")
    parser.add_argument("--class_num", type=str, default=46, help="The class number of object")
    parser.add_argument("--input_size",type=int, default="380", help=None)
    parser.add_argument("--batch_size", type=int, default=20, help="The number of data that input model")
    parser.add_argument("--model_dir", type=str, default="dataset/model/efficientnet-b4.pth")
    parser_arg = parser.parse_args()
    model = torch.load(parser_arg.model_dir, map_location=torch.device('cpu'))
    # evaluate
    criterion = nn.CrossEntropyLoss()
    if use_gpu:
        model = model.cuda()
        criterion = criterion.cuda()
    print('-' * 10)
    print('inference:')
    # 加载数据集

    outResult = inference(parser_arg, model, criterion)
    print(outResult)
