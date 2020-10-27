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
import parameters  as params

from dataset import MakeTrainValidSet, MapillaryDataset
from model.model import DeeplabV3
# from utils import add_weight_decay
# from undistort import distort_augmenter
from utils_losses import DiceLoss, CE_DiceLoss, CrossEntropyLoss2d, LovaszSoftmax
# from lr_scheduler import Poly
# from torchvision.utils import make_grid
# from utils_metrics import Evaluator

def transformsFunction():
    train_seq = iaa.Sequential([iaa.size.Resize({"height": params.img_h, "width": params.img_w}, interpolation='nearest'),
                                iaa.Fliplr(0.5),
                                iaa.Multiply((0.8, 1.5)),                               
                                iaa.OneOf([ 
                                            iaa.MotionBlur(k=4),
                                            iaa.GaussianBlur((1.0)),
                                            iaa.Noop(),
                                        ]),
                                iaa.Sometimes(0.5, iaa.Sharpen(alpha=(0, 0.8))),
                                # iaa.Sometimes(0.3,  iaa.PiecewiseAffine(scale=(0.01, 0.05))),
                                ])
                                
    valid_seq = iaa.Sequential([iaa.size.Resize({"height": params.img_h, "width": params.img_w}, interpolation='nearest'),
                                ])
    return train_seq, valid_seq

def worker_init_fn(worker_id):
    imgaug.seed(np.random.get_state()[1][0] + worker_id)


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.exists("./Mapillary_training.json"):
        MakeTrainValidSet(data_dir)
    else:
        with open("./Mapillary_training.json", "r") as f:
            trainSet = json.load(f)
        with open("./Mapillary_validation.json", "r") as f:
            validSet = json.load(f)
    
    print("trainingset: {}, validset: {}".format(len(trainSet), len(validSet)))
    
    train_seq, valid_seq = transformsFunction()

    train_dataset = MapillaryDataset(trainSet, seq=train_seq)
    val_dataset = MapillaryDataset(validSet, seq=valid_seq)

    num_train_batches = int(len(train_dataset)/params.batch_size)
    num_val_batches = int(len(val_dataset)/params.batch_size)

    print ("num_train_batches:", num_train_batches)
    print ("num_val_batches:", num_val_batches)

    train_loader = DataLoader(dataset     = train_dataset,
                            batch_size    = params.batch_size,
                            shuffle       = True,
                            num_workers   = 4,
                            worker_init_fn= worker_init_fn)
    
    val_loader = DataLoader(dataset     = val_dataset,
                            batch_size  = params.batch_size,
                            shuffle     = False,
                            num_workers = 0)
    
    seg_cls = train_dataset.get_num_cls()
    print("num_classes: ", seg_cls)

    '''
        mode == 0: ResNet18_OS16()
        mode == 1: ResNet34_OS16()
        mode == 2: ResNet18_OS8()
        mode == 3: ResNet34_OS8()
        mode == 4: ResNet50_OS16()
        mode == 5: ResNet101_OS16()
        mode == 6: ResNet152_OS16()
    '''

    model_id = "FE4_CED_SGD"
    model = DeeplabV3(model_id, "./", seg_cls, FeatureExtractor=4)
    model.to(device)

    class_weights = None
    # if os.path.exists("./{}_ClassWeights_Mapillary.pkl".format(seg_cls)):
    #     with open("./{}_ClassWeights_Mapillary.pkl".format(seg_cls), "rb") as file:
    #         class_weights = np.array(pickle.load(file))

    #     class_weights = torch.from_numpy(class_weights) 
    #     class_weights = class_weights.type(torch.FloatTensor).to(device)

    model_dir = model.module.model_dir if hasattr(model, 'module') else model.model_dir
    ckpt_path = model.module.checkpoints_dir if hasattr(model, 'module') else model.checkpoints_dir

    ''' load weights '''
    isLoad = False
    init_epoch = 0
    if isLoad:
        print("Load weights !!")
        ckpt = torch.load(ckpt_path + "/last.pth")
        model.load_state_dict(ckpt['model_state_dict'])
        init_epoch = ckpt["epoch"]

    trainable_params = filter(lambda p:p.requires_grad, model.parameters())

    params = {
        "lr": params.learning_rate,
        "momentum": params.momentum,
        "nesterov": params.nesterov,
    }

    # optimizer = torch.optim.Adam(params=trainable_params, **params)
    ptimizer = torch.optim.SGD(params=trainable_params, **params)

    # criterion = CrossEntropyLoss2d(weight=class_weights, reduction="mean")
    # criterion = nn.CrossEntropyLoss(weight=class_weights)
    # criterion = DiceLoss()
    criterion = CE_DiceLoss(reduction="mean", weight=class_weights)
    # criterion = LovaszSoftmax()


