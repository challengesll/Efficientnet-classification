# 分类
A demo for train your own dataset on EfficientNet

Thanks for the >[A PyTorch implementation of EfficientNet](https://github.com/lukemelas/EfficientNet-PyTorch), I just simply demonstrate how to train your own dataset based on the EfficientNet-Pytorch.

## 介绍

应社会发展的需要，垃圾分类不仅成为时代的潮流也成为与自然环境共生的一种生活素养，为解决人们对垃圾类别的困扰，本项目通过训练一个垃圾分类器，将241项类别的垃圾划分为4个类，分别是厨余垃圾，可回收物，其他垃圾和有害垃圾。

## Step 1：Prepare your own classification dataset
---
Then the data directory should looks like:   
```
-dataset\
    -model\
    -train\
        -厨余垃圾\
        -其他垃圾\
        ...
    -test\
        -厨余垃圾\
        -其他垃圾\
        ...
```

## Step 2: train and test 
(1)You can choose to download the pre-trained model automatically or not by modify the ```line 169```.

The pre-trained model is available on >[release](https://github.com/lukemelas/EfficientNet-PyTorch/releases). 

You can download them under the folder ```eff_weights```.

(2)训练垃圾分类的模型.
 ```python
    python3 train.py
 ```
(3)You can get the final results and the best model on ```dataset/model/```.

(4)测试模型

```python
python3 evaluate.py
```

(5)模型的推理

```python 
python3 inference.py
```

