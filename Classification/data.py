import PIL
import cv2
import numpy as np
import torch.nn as nn
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
class CustomDataset(Dataset):
    def __init__(self, label_file_path):
        with open(label_file_path, 'r') as f:
            self.imgs = list(map(lambda line: line.strip().split(' '), f))
            self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.Resize((32,32)),
            transforms.ToTensor()])
            
    def __getitem__(self, index):
        path, label = self.imgs[index]
        img = PIL.Image.open(path).convert('RGB')
        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        img = self.transform(img)
        label = int(label)
        return img, label
    def __len__(self): #这个函数也必须要写，它返回的是数据集的长度，也就是多少张图片，要和loader的长度作区分
        return len(self.imgs)

