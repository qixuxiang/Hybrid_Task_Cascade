#-*- coding:utf-8 -*-
import os
import time
import datetime
#import mmcv
import cv2
import json
import numpy as np
import pycocotools.mask as maskutil
import pycocotools.coco as COCO
from itertools import groupby
from skimage import measure,draw,data
from PIL import Image
import matplotlib.pyplot as plt

data_path = './jinnan2_round2_train_20190401/'

if not os.path.exists(data_path + 'label'):
    os.makedirs(data_path + 'label')

current_path = './'

#if not os.path.exists(current_path + 'npydata'):
#    os.makedirs(current_path + 'npydata')
#
#if not os.path.exists(current_path + 'data'):
#    os.makedirs(current_path + 'data')

#if not os.path.exists(current_path + 'pre_train'):
#    os.makedirs(current_path + 'pre_train')

def get_index(image_id,load_dict):#get seglist and label list by image_id
    seg_list = []
    label_list = []
    for i in range(len(load_dict['annotations'])):
        if image_id == load_dict['annotations'][i]['image_id']:
            seg_list.append(i)
            label_list.append(load_dict['annotations'][i]['category_id'])
    return seg_list,label_list

def get_color(class_id):#for Distinguish different classes
    return class_id

with open(data_path + 'train_restriction.json','r') as f:
    load_dict = json.load(f)
    paths = os.listdir(data_path + 'restricted')
    for im_path in paths:
        im = cv2.imread(data_path + 'restricted/'+ im_path)
        seg_list,label_list = get_index(int(im_path[:-4]),load_dict)
        #print(seg_list)
        #print(label_list)
        #masks = np.zeros((im.shape[0],im.shape[1], 1), np.uint8)
        seg = []
        masks = []
        cnt = 0
        for seg_idx in seg_list:
            seg = load_dict['annotations'][seg_idx]['segmentation'][0] #load first seg in seg list
            compactedRLE = maskutil.frPyObjects([seg], im.shape[0], im.shape[1]) #compress through RLE
            mask = maskutil.decode(compactedRLE) #decode to mask

            mask=np.reshape(mask,(im.shape[0],im.shape[1])) #for display
            mask = mask*get_color(label_list[cnt]) #change color for different class
            masks.append(mask) #add sub mask for a full mask

            cnt+=1
        final_mask = np.zeros((im.shape[0],im.shape[1]), np.uint8) #final mask for each img
        for mask in masks: #merge all mask into final mask
            mask[final_mask!=0]=0
            final_mask = final_mask + mask
        #plt.imshow(final_mask) #show final mask

        cv2.imwrite(data_path +'label/'+ im_path.replace('jpg','png'), final_mask)
        #plt.show()
