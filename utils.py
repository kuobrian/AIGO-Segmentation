import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import Mapillary_labels

def draw_label(img):
    img_height, img_width = img.shape

    img_color = np.zeros((img_height, img_width, 3), dtype=np.uint8)

    for label in Mapillary_labels:
        obj_isinstance = label.hasInstanceignoreInEval
        color = label.color
        name = label.name
        trainId = label.trainId

        if obj_isinstance:
            img_color[img == trainId] = color
        
        ''' Road class '''
        if name in ['Curb', 'Road', 'Lane Marking - Crosswalk',
                    "Lane Marking - General", 'Sidewalk',  'Rail Track',
                    'Bike Lane', 'Pedestrian Area', 'Crosswalk - Plain', 'Parking', 'Guard Rail', 'Barrier']:
            img_color[img == trainId] = [128, 64, 128]
        
        ''' Building class '''
        if name in ['Wall', 'Building']:
            img_color[img == trainId] = [70, 70, 70]

        if name in ['Vegetation', 'Terrain']:
            img_color[img == trainId] = [107, 142, 35]
        
        if name in ['Sky', 'Fence']:
            img_color[img == trainId] = color
        
        # unique_img = np.unique(img)
        # if not obj_isinstance and name not in ['Curb', 'Road', 'Lane Marking - Crosswalk',
        #             "Lane Marking - General", 'Sidewalk',  'Rail Track',
        #             'Bike Lane', 'Pedestrian Area', 'Crosswalk - Plain', 'Wall', 'Building',
        #               'Vegetation', 'Sky', 'Fence', 'Terrain', 'Parking', 'Guard Rail', 'Barrier']:
        #     img_color[img == trainId] = color
        #     if trainId in unique_img:
        #         print("******", name)
            

        
    return img_color


def decodeImgs(label_imgs):
    decode_imgs = []
    for label_img in label_imgs:
        img_color = draw_label(label_img)
        
        decode_imgs.append(torch.from_numpy(np.transpose(img_color, (2, 0, 1))))
        
    return decode_imgs


def denormalize(images, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    for i in range(images.shape[1]):
        images[:, i, :, :] = images[:, i, :, :] * std[i] + mean[i]       
    
    images = images[:, [2, 1, 0], :, :]
    images = images * 255
    return images.byte()

def add_weight_decay(net, l2_value, skip_list=()):
    # https://raberrytv.wordpress.com/2017/10/29/pytorch-weight-decay-made-easy/

    decay, no_decay = [], []
    for name, param in net.named_parameters():
        if not param.requires_grad:
            continue # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)

    return [{'params': no_decay, 'weight_decay': 0.0}, {'params': decay, 'weight_decay': l2_value}]


class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        MIoU = np.nanmean(MIoU)
        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)



