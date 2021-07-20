# _*_ conding: utf-8 _*_
import os
import numpy as np
import random
import cv2
"""
将垃圾分类的数据集划分为训练集，验证集和测试集，比例分别为8:1:1
train
test
val
"""
def save_data(root_path, images_path, data, data_class, data_type='train'):
    for image in data:
        if image.split('.')[1] not in ['jpeg','jpg','png']:
            continue
        image_path = os.path.join(images_path, image)
        out_path = os.path.join(root_path, data_type, data_class)
        print(out_path)
        if not os.path.exists(out_path):
            print('kkkkkkk')
            os.makedirs(out_path)
        if image.endswith('.jpeg'):
            img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            cv2.imencode('.jpeg',img)[1].tofile(os.path.join(out_path, image))
        else:
            try:
                img = cv2.imread(image_path)
                cv2.imwrite(os.path.join(out_path, image), img)
            except:
                continue

        # cv2.imwrite(os.path.join(out_path, image), img)

if __name__ == "__main__":
    file_path = "../dataset"
    dataset_name = 'brand'
    train_radio = 0.8
    test_radio = 0.1
    # 列出数据集中的文件类型
    trash_catagories = os.listdir(os.path.join(file_path, dataset_name))
    num_catagories = len(trash_catagories)
    print(num_catagories)
    '''
    划分的规则可以使用以下两种规则：
    1. 按照大的类别进行划分，比如从可回收垃圾中划分0.8作为训练集，0.1测试集，0.1验证集
    2. 按照小的分类进行划分，比如从可回收垃圾_塑料纸杯0.8训练集，0.1测试集，0.1验证集
    以下是按照小的分类进行划分
    '''
    for classname in trash_catagories:
        # 每个分类中的图像
        #print(classname)
        images_list = os.listdir(os.path.join(file_path, dataset_name, classname))
        #print('hhhhh',images_list)
        train_num = int(len(images_list) * train_radio)
        print(train_num)
        test_num = int(len(images_list) * test_radio)
        # 随机打乱数据
        # if classname != '':
        #     continue
        random.shuffle(images_list)
        train_data = images_list[:train_num]
        test_data = images_list[train_num:train_num + test_num]
        val_data = images_list[train_num+test_num:]

        save_data(file_path, os.path.join(file_path,dataset_name,classname),train_data, classname, 'train')
        print('jjjjjjjjjjjj')
        save_data(file_path, os.path.join(file_path, dataset_name, classname), test_data, classname, 'test')
        save_data(file_path, os.path.join(file_path, dataset_name, classname), val_data, classname, 'val')


