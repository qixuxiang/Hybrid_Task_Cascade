import os
import numpy as np
import csv
import glob
import mmcv
import pycocotools.mask as maskUtils
import torch

from mmdet.core import get_classes
from mmdet.core import results2json, coco_eval
from mmdet.datasets import to_tensor
from mmdet.datasets.transforms import ImageTransform
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.apis import inference_detector, show_result

if not os.path.exists('submit_result'):
    os.makedirs('submit_result')

if not os.path.exists('test_png'):
    os.makedirs('test_png')

res_img_pth = './restrict.csv'
res_file = open(res_img_pth, 'r')
content = res_file.read()
res_images = []
rows = content.split('\n')
res_images.append(str(rows[0][1:]))
for row in rows[1:]:
    res_images.append(str(row))
print('------------------------------------------------------')

# 构建网络，载入模型                                                                        
cfg = mmcv.Config.fromfile('./configs/cascade_mask_rcnn_x101_32x4d_fpn_1x.py')                   
cfg.model.pretrained = None                                                                 
model = build_detector(cfg.model, test_cfg=cfg.test_cfg)                                    
#model = build_detector(cfg.model)                                    
_ = load_checkpoint(model, './model_dir/cascade/epoch_36.pth')   

def npy_generator(image,
                  dataset='jinnan',
                  score_thr=0.5,
                  with_mask=True):
    # 读取文件名和imread读取图像
    basename = os.path.basename(image)
    img_name = basename[:-4]
    img = mmcv.imread(image)
    # infer单图
    result = inference_detector(model, img, cfg)
    if with_mask:
        bbox_result, segm_result = result
    else:
        bbox_result=result
    if isinstance(dataset, str):#  add own data label to mmdet.core.class_name.py
        class_names = get_classes(dataset)
        # print(class_names)
    elif isinstance(dataset, list):
        class_names = dataset
    else:
        raise TypeError('dataset must be a valid dataset name or a list'
                        ' of class names, not {}'.format(type(dataset)))
    h, w, _ = img.shape
    img_show = img[:h, :w, :]
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)
    bboxes = np.vstack(bbox_result)
    # 定义全零mask
    masks_1 =  np.zeros((h, w))
    masks_2 =  np.zeros((h, w))
    masks_3 =  np.zeros((h, w))
    masks_4 =  np.zeros((h, w))
    masks_5 =  np.zeros((h, w))

    if score_thr > 0:
        assert bboxes.shape[1] == 5
        scores = bboxes[:, -1]
        inds = scores > score_thr
        bboxes = bboxes[inds, :]
        labels = labels[inds]
    
    print(len(res_images[:-1])) # res_images最后一个元素是''
    if basename in res_images[:-1]:
        segms = mmcv.concat_list(segm_result)
        inds = np.where(bboxes[:, -1] > score_thr)[0]
        print("inds.shape",inds.shape)
        for i in inds:
            color_mask = np.random.randint(0, 256, (1, 3), dtype=np.uint8)
            mask_binary = maskUtils.decode(segms[i])#每个mask
            mask = mask_binary.astype(np.bool)
            img_show[mask] = img_show[mask] * 0.5 + color_mask * 0.5
            indx = class_names.index(class_names[labels[i]]) + 1 #index(tieke) = 1
            if(indx) == 1:
                masks_1 = np.array(masks_1 + mask)
            elif(indx) == 2:                                                                     
                masks_2 = np.array(masks_2 + mask)            
            elif(indx) == 3:                                                                   
                masks_3 = np.array(masks_3 + mask)
            elif(indx) == 4:                                                                  
                masks_4 = np.array(masks_4 + mask)
            elif(indx) == 5:                                                                   
                masks_5 = np.array(masks_5 + mask)
            else:
                pass
    
    masks_1 = np.array(masks_1 >= 1).astype(np.uint8)
    masks_2 = np.array(masks_2 >= 1).astype(np.uint8)
    masks_3 = np.array(masks_3 >= 1).astype(np.uint8)
    masks_4 = np.array(masks_4 >= 1).astype(np.uint8)
    masks_5 = np.array(masks_5 >= 1).astype(np.uint8)

    #定义mask保存路径
    masks_1_pth =  './submit_result/' + img_name + '_1'
    masks_2_pth =  './submit_result/' + img_name + '_2'
    masks_3_pth =  './submit_result/' + img_name + '_3'
    masks_4_pth =  './submit_result/' + img_name + '_4'
    masks_5_pth =  './submit_result/' + img_name + '_5'
    #保存mask文件
    np.save(masks_1_pth, masks_1)
    np.save(masks_2_pth, masks_2)
    np.save(masks_3_pth, masks_3)
    np.save(masks_4_pth, masks_4)
    np.save(masks_5_pth, masks_5)


    result_img=mmcv.imshow_det_bboxes(
        img_show,
        bboxes,
        labels,
        class_names=class_names,
        score_thr=score_thr,
        show=False,
        out_file = './test_png/' + basename)
    return  result_img


def main():
    '''
    # 测试一张图片
    image =  './jinnan/jinnan2_round2_test_a_20190401/1110.jpg'
    npy_generator(image,score_thr=0.6,with_mask=True)
    '''
    infer_path = './jinnan/jinnan2_round2_test_a_20190401'
    images = glob.glob('{}/*.jpg'.format(infer_path))
    for image in images:
        npy_generator(image, score_thr = 0.5, with_mask = True)
    #mmcv.imwrite(answer,"1.png")
    
if __name__ == '__main__':
    main()
