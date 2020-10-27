import os
import numpy as np
import cv2
import json
import torch
import imgaug
import pickle
import matplotlib
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import DataLoader
from imgaug import augmenters as iaa

import parameters  as params

from dataset import MakeTrainValidSet, MapillaryDataset
from model.backbone_model import DeeplabV3
from utils import Evaluator


def GetTrainedModel(ckpt_path = None, num_cls=None, mode=1):
    ckpt = torch.load(ckpt_path)
    print("weight's epoch: {}".format(ckpt["epoch"]))
    
    model = DeeplabV3("eval", "./", num_cls, FeatureExtractor=mode)
    model.load_state_dict(ckpt['model_state_dict'])

    return model

if __name__ == "__main__":
    with open("./Mapillary_validation.json", "r") as f:
        validSet = json.load(f)
                                
    valid_seq = iaa.Sequential([iaa.size.Resize({"height": params.img_h, "width": params.img_w}, interpolation='nearest'),
                            ])

    
    val_dataset = MapillaryDataset(validSet, seq=valid_seq)

    val_loader = DataLoader(dataset     = val_dataset,
                            batch_size  = params.batch_size,
                            shuffle     = False,
                            num_workers = 0)
    
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seg_cls = val_dataset.get_num_cls()

    folder = "FE4_CED_SGD"
    folder = params.model_id
    ckpt_path = './logs/{}/checkpoints/last.pth'.format(folder)
    ckpt_path = './logs/{}/checkpoints/best_train.pth'.format(folder)
    ckpt_path = './logs/{}/checkpoints/best_fwiou.pth'.format(folder)
    ckpt_path = './logs/{}/checkpoints/best_pixacc.pth'.format(folder)
    ckpt_path = './logs/{}/checkpoints/best_valid.pth'.format(folder)

    model = GetTrainedModel(ckpt_path = ckpt_path,
                            num_cls   = seg_cls,
                            mode      = 4)
    
    model.to(device)
    model.eval()

    evaluator = Evaluator(seg_cls)

    for img, label in val_loader:
        print(img.shape)
        img = img.to(device, dtype=torch.float)
        label = label.to(device, dtype=torch.long)
        
        pred_raw = model(img)
        pred_raw = pred_raw.data.cpu().numpy()
        pred_label_imgs = np.argmax(pred_raw, axis=1) # (shape: (batch_size, img_h, img_w))

        label = label.data.cpu().numpy()
        evaluator.add_batch(label, pred_label_imgs)

    Acc = evaluator.Pixel_Accuracy()
    Acc_class = evaluator.Pixel_Accuracy_Class()
    mIoU = evaluator.Mean_Intersection_over_Union()
    FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()

    print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))


        