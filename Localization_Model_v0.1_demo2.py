#!/usr/bin/env python
# coding: utf-8

# On final dataset, using only labeled images ; Resnet18 -> Linear

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch.utils.data import Dataset
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
from PIL import Image # pip3 install pillow
import random
import cv2
import time
import os



img_folder = "images/"
imlist = []
imlist =[os.path.join(img_folder, f) for f in os.listdir(img_folder) 
         if os.path.isfile(os.path.join(img_folder, f))]


df_test = pd.DataFrame(data=imlist, columns=["image"])
df_test['points'] = [0 for _ in imlist]


# ### Image Transform

class ResizeAspect(object):
    def __init__(self, h, w):
        self.hw = (h, w)
        self.rescale_factor=None
        self.shift_h=None
        self.shift_w=None
        
    def do_image(self, img):
        h, w = self.hw
        img_h, img_w = img.shape[0], img.shape[1]
        rescale_factor = min(w/img_w, h/img_h)
        new_w = int(img_w * rescale_factor)
        new_h = int(img_h * rescale_factor)
        resized_image = cv2.resize(img, (new_w,new_h), interpolation = cv2.INTER_CUBIC)

        canvas = np.full((h, w, 3), 128, dtype=np.uint8)
        shift_h = (h-new_h)//2
        shift_w = (w-new_w)//2
        canvas[(h-new_h)//2:(h-new_h)//2 + new_h,(w-new_w)//2:(w-new_w)//2 + new_w,  :] = resized_image
        img = canvas.copy()
        self.rescale_factor=rescale_factor
        self.shift_h = shift_h
        self.shift_w = shift_w
        return img
    
    def do_box(self, box):
        box = box.reshape(-1,2)
        box *=self.rescale_factor
        box[: ,0] += self.shift_w
        box[: ,1] += self.shift_h
        box = box.reshape(-1)
        return box
    
    def undo_box(self, box):
        box = box.reshape(-1,2)
        box[: ,0] -= self.shift_w
        box[: ,1] -= self.shift_h
        box /=self.rescale_factor
        box = box.reshape(-1)
        return box


class FinalTransform:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])
    
    def transform_inv(self,img):
        inp = img.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        return inp


# ### Dataset Loader


class LicenseDataset(Dataset):
    def __init__(self, df):
        
        self.imgs = list(df.image)
        self.boxes = df.points.tolist()
        self.final_transform = FinalTransform()
        self.transform = self.final_transform.transform
        self.transform_inv = self.final_transform.transform_inv
        self.resizer = ResizeAspect(h=224, w=224)
            
    def __getitem__(self, index):
        path= self.imgs[index]
        box = self.boxes[index]
        
        
        img = Image.open(path).convert('RGB')
        img = self.resizer.do_image(np.array(img))
        img = self.final_transform.transform(img)
#         box = self.resizer.do_box(box)
        box = np.array(box, dtype=np.float32)

        factor = np.array(
            [self.resizer.rescale_factor, self.resizer.shift_h, self.resizer.shift_w],
            dtype=np.float32
        )
        return img, box, factor, index
      
    
    def __len__(self):
        return len(self.imgs)


test = LicenseDataset(df_test)

test_loader = torch.utils.data.DataLoader(
                test, batch_size=32,shuffle=False,
                num_workers=4, pin_memory=True)


# ### Defining Model

model = models.resnet18(pretrained=True)
'''
output of our model is :
x1, y1,
x2, y2,
x3, y3,
x4, y4,
conf -> only when no bounding box images are taken
'''
num_feature = model.fc.in_features
num_output = 8#9
model.fc = nn.Linear(num_feature, num_output)
model = model.cpu()

model.load_state_dict(torch.load('saved_states/model_state_v0.pth', map_location='cpu'))


test_batch = 0
indx = -1
model.eval()
print('Working in test mode')

resizer = test.resizer
for j,(ims, boxes, factors, index) in enumerate(test_loader):

    index = index.cpu().numpy()
    inputs = ims.cpu()
    factors = factors.cpu().numpy()
    outputs = model(inputs)
    outputs = outputs.data.cpu().numpy()

    for indx in range(len(outputs)):
        loader_indx = index[indx]
        factor = factors[indx]

        path= test.imgs[loader_indx]
        img = Image.open(path).convert('RGB')
        resizer.rescale_factor=factor[0]
        resizer.shift_h=factor[1]
        resizer.shift_w=factor[2]

        out = outputs[indx]
        out = resizer.undo_box(out)
        out = np.append(out, out[:2]).reshape(-1,2)
        
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        for i in range(len(out)-1):
            img = cv2.line(img, tuple(out[i]), tuple(out[i+1]), color=(0,255,100), thickness=2)
        cv2.imwrite(path.replace("images", "prediction"), img)
print('finished prediction')

