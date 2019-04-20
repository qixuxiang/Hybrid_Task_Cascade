#coding=utf-8
import json
import numpy as np
from PIL import Image
import os
import argparse

parser = argparse.ArgumentParser(description='path of test json')
parser.add_argument('--file_path', help='test set file path')
parser.add_argument('--json_path', help='test set json_path')
args = parser.parse_args()


image_list=os.listdir(args.file_path)
data=dict()
data["licenses"]=[{"name":"Attribution-NonCommercial-ShareAlike License","url":"","id":1}]
data["info"]={"data_created":"now"}
data["images"]=[]
data["annotations"]=[]
num=1
for ima_file in image_list:
    print("processing",ima_file)
    information=dict()
    ann=dict()
    information["license"]=1
    img=np.array(Image.open(args.file_path+"/"+ima_file))
    h,w=img.shape[0],img.shape[1]
    information["height"]=h
    information["data_captured"]=""
    information["width"]=w
    information["file_name"]=ima_file
    information["coco_url"]=""
    information["flickr_url"]=""
    information["id"]=num
    data["images"].append(information)
    ann["bbox"]=[]
    ann["minAreaRect"]=[]
    ann["area"]=[]
    ann["category_id"]=""
    ann["iscrowd"]=0
    ann["id"]=num
    ann["image_id"]=num
    data["annotations"].append(ann)
    num+=1
data["categories"]=[{"name": "铁壳打火机", "supercategory": "restricted_obj", "id": 1}, {"name": "黑钉打火机", "supercategory": "restricted_obj", "id": 2}, {"name": "刀具", "supercategory": "restricted_obj", "id": 3}, {"name": "电源和电池", "supercategory": "restricted_obj", "id": 4}, {"name": "剪刀", "supercategory": "restricted_obj", "id": 5}]
with open(args.json_path,"w") as f:
    json.dump(data,f)