#coding=utf-8
import os
import xml.etree.ElementTree as ET
# #打开xml文档
# p='E:\BaiduNetdiskDownload\VOC07+12+test\VOCdevkit\VOC2007\Annotations'
p='/home/data/zhq/VOC2007/Annotations'
a_p = os.listdir(p)
a_p_a = [os.path.join(p,i) for i in a_p]
tv = []
for i in a_p_a:
    if i[-3:] == 'xml':
        in_file = open(i, encoding='utf-8')
        tree = ET.parse(in_file)
        root = tree.getroot()
        temp = []
        for obj in root.iter('object'):
            temp.append(obj.find('name').text)
        tv.append(temp)
            # if obj.find('name').text == 'tvmonitor':
                        #     tv.append(i)

for i in tv:
    if 'tvmonitor' in i:
        print(i)
#
# p1 = '/home/zhq/paperCode/faster-rcnn-pytorch-master/data_split/voc_incre_classes_1_nb_val.txt'
# image_ids_1 = open(p1).read().strip().split()
# image_ids = []
# for i in image_ids_1:
#     if i[-1] == 'g':
#         temp = i.split('/')[-1][:-4]
#         image_ids.append(temp)
# print(image_ids)

# tv_ids = []
# for i in tv:
#     temp = i.split('/')[-1][:-4]
#     tv_ids.append(temp)
#
# print(tv_ids)

import cv2
def draw_gt_pre():
    picture_path = '/home/zhq/paperCode/faster-rcnn-pytorch-master/map_out/b19_n1/directy_test_n1/images-optional'
    gt_path = '/home/zhq/paperCode/faster-rcnn-pytorch-master/map_out/b19_n1/directy_test_n1/ground-truth'
    pre_path = '/home/zhq/paperCode/faster-rcnn-pytorch-master/map_out/b19_n1/directy_test_n1/detection-results'

    img_list = os.listdir(picture_path)
    all_img = [os.path.join(picture_path,img) for img in img_list]


    # picture = cv2.imread(picture_path)		# picture_path为图片路径;(cv读取的文件为BGR形式)

    # cv2.rectangle(picture, (x_min,y_min), (x_max,y_max), (255, 0, 255), -1)
