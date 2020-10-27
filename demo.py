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
from torchvision import transforms
from torch.utils.data import DataLoader
from imgaug import augmenters as iaa
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid

import parameters  as params

from dataset import Mapillary_labels
from model.backbone_model import DeeplabV3
from lr_scheduler import Poly
from utils import add_weight_decay, Evaluator, draw_label, denormalize
from utils_losses import DiceLoss, CE_DiceLoss, CrossEntropyLoss2d, LovaszSoftmax


def GetTrainedModel(ckpt_path = None, num_cls=None, mode=1):
    ckpt = torch.load(ckpt_path)
    print("weight's epoch: {}".format(ckpt["epoch"]))
    
    model = DeeplabV3("eval", "./", num_cls, FeatureExtractor=mode)
    model.load_state_dict(ckpt['model_state_dict'])

    return model

def DemoSingleImage(model, img_path, img_tfs, device='cuda'):
    img = cv2.imread(img_path, -1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_rgb = img_rgb / 255.0
    img_rgb = img_rgb.astype(np.float32)
    transform_img = torch.unsqueeze(img_tfs(img_rgb), 0)
    transform_img = transform_img.to(device, dtype=torch.float)

    print(transform_img.shape)


    pred_raw = model(transform_img)
    pred_raw = pred_raw.data.cpu().numpy()
    pred_label_imgs = np.argmax(pred_raw, axis=1)
    for idx, label in enumerate(Mapillary_labels):
        if label.id in np.unique(pred_label_imgs):
            print(label.trainId, " => ", label.name)


    pred_label_imgs = pred_label_imgs.astype(np.uint8)[0]
    pred_label_img_color = draw_label(pred_label_imgs)


    print(pred_label_img_color.shape)
    resize_vis = cv2.resize(img, (params.img_w, params.img_h))


    overlayed_img = 0.35*resize_vis + 0.65*pred_label_img_color
    overlayed_img = overlayed_img.astype(np.uint8)

    combined_img = np.zeros((2*params.img_h, 2*params.img_w, 3), dtype=np.uint8)
    combined_img[0:params.img_h, 0:params.img_w] = resize_vis
    combined_img[0:params.img_h, params.img_w:(2*params.img_w)] = pred_label_img_color
    combined_img[params.img_h:(2*params.img_h), (int(params.img_w/2)):(params.img_w + int(params.img_w/2))] = overlayed_img

    cv2.imshow("combined_img", combined_img)
    cv2.waitKey(0)
    



if __name__ == "__main__":
                                
    img_seq = iaa.Sequential([iaa.size.Resize({"height": params.img_h, "width": params.img_w}, interpolation='nearest'),
                            ])    
    
    img_tfs = transforms.Compose([img_seq.augment_image,
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                    std=(0.229, 0.224, 0.225))
                        ])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seg_cls = 66

    folder = "DeepLab_m4_M_CED_SGD"

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

    DemoSingleImage(model, img_path='./testimg.jpg', img_tfs=img_tfs)
    

    # video_paths = [
    #             "./exp/Test3.mp4",
    #             "./exp/20200806_experiment1.avi",
    #             "./exp/20200806_experiment2.avi"]

    # for video_path in video_paths:
    #     print("Predict: ", video_path)
    #     ckpt_name = os.path.splitext(ckpt_path.split("/")[-1])[0]
    #     save_path = './exp/{}_{}_{}.avi'.format(os.path.splitext(video_path)[0].split('/')[-1], folder, ckpt_name)
    #     # if not os.path.exists(save_path):
    #     infer_video(model, video_tfs, device, video_path, save_path)
    # assert(0)



