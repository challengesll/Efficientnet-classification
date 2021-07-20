# _*_ coding: utf-8 _*_
from torch.utils.data import dataset
import os
import numpy as np
import cv2
class SquDataset(dataset):
    """
    覆盖一个dataset类，主要实现将所有的图像序列加载进来，并输入到模型进行训练

    """
    def __init__(self, data_path, split='train', width=380, height=256, transfunc=None):
        """
        需要将所有的图像加载进来,先将一个图像的大小转化为（width，height，3）
        加载每一个instance
        """
        self.width = width
        self.height = height
        self.transfunc = transfunc
        folder = os.path.join(data_path, split)
        self.fnames, labels = [], []
        for label in sorted(os.listdir(folder)):  # dataset/amap_pro/train
            for fname in os.listdir(os.path.join(folder, label)):
                self.fnames.append(os.path.join(folder, label, fname))
                labels.append(label)
        assert len(labels) == len(self.fnames)
        print('Number of {} videos: {:d}'.format(split, len(self.fnames)))

        self.label_dic = {label : index for index, label in enumerate(sorted(set(labels)))}
        self.labels = np.array([self.label_dic[label] for label in labels], dtype=int)

        if not os.path.exists('./../dataset/model/amap_labels.txt'):
            with open('../dataset/model/amap_labels.txt', 'w') as f:
                for id, label in enumerate(sorted(self.label_dic)):
                    f.writelines(str(id) + ' ' + label + '\n')

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):
        """
        加载图像并返回每次执行
        """

        frames = self.load_frame(self.fname[index])
        label = self.labels[index]
        return frames, label

    def load_frame(self, fname):
        """
        加载每个图像序列，并resize图像大小
        """
        frames = sorted([os.path.join(fname, img) for img in os.listdir(fname)])
        frame_count = len(frames)   # 每个图像序列的长度
        buffer = np.empty((frame_count, self.resize_height, self.resize_width, 3), np.dtype('float32'))
        for i, frame_name in enumerate(frames):
            img = self.transfunc(cv2.imread(frame_name)).float()
            try:

                frame = np.array(img).astype(np.float64)            # 默认是图像的大小是一样的
                buffer[i] = frame
            except:
                print("输入的帧大小不同")

        return buffer