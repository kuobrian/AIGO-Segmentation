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
from lr_scheduler import Poly
from utils import add_weight_decay, Evaluator
from utils_losses import DiceLoss, CE_DiceLoss, CrossEntropyLoss2d, LovaszSoftmax


# from undistort import distort_augmenter
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

    model = DeeplabV3(params.model_id, "./", seg_cls, FeatureExtractor=4)
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

    paramsOptim = {
        "lr": params.learning_rate,
        "momentum": params.momentum,
        "nesterov": params.nesterov,
    }

    # optimizer = torch.optim.Adam(params=trainable_params, **params)
    optimizer = torch.optim.SGD(params=trainable_params, **paramsOptim)

    # criterion = CrossEntropyLoss2d(weight=class_weights, reduction="mean")
    # criterion = nn.CrossEntropyLoss(weight=class_weights)
    # criterion = DiceLoss()
    criterion = CE_DiceLoss(reduction="mean", weight=class_weights)
    # criterion = LovaszSoftmax()

    num_epochs = init_epoch + params.epochs
    iters_per_epoch = len(train_loader)

    lr_scheduler = Poly(optimizer, num_epochs, iters_per_epoch)
    
    ''' Plot lr schedule '''
    # y = []
    # for epoch in range(init_epoch, num_epochs):
    #     lr_scheduler.step()
    #     print('Epoch-{0} lr: {1}'.format(epoch, optimizer.param_groups[0]['lr']))
    #     y.append(optimizer.param_groups[0]['lr'])
    # plt.plot(y, '.-', label='LambdaLR')
    # plt.xlabel('epoch')
    # plt.ylabel('LR')
    # plt.tight_layout()
    # plt.savefig('./LR.png', dpi=300)

    epoch_losses_train = []
    epoch_losses_val = []
    mertics_val_miou = []
    mertics_val_pixAcc = []
    mertics_val_FWIoU = []

    if isLoad and os.path.exists("%s/epoch_losses_val.pkl" % model_dir) and\
        os.path.exists("%s/epoch_losses_train.pkl" % model_dir):

        with open("%s/epoch_losses_train.pkl" % model_dir, "rb") as file:
            epoch_losses_train = list(np.array(pickle.load(file)))

        with open("%s/epoch_losses_val.pkl" % model_dir, "rb") as file:
            epoch_losses_val = list(np.array(pickle.load(file)))

        with open("%s/epoch_miou_val.pkl" % model_dir, "rb") as file:
            mertics_val_miou = list(np.array(pickle.load(file)))

        with open("%s/epoch_FWIoU_val.pkl" % model_dir, "rb") as file:
            mertics_val_FWIoU = list(np.array(pickle.load(file)))
        
        with open("%s/epoch_pixAcc_val.pkl" % model_dir, "rb") as file:
            mertics_val_pixAcc = list(np.array(pickle.load(file)))

        print("Load Prev train Loss: ", len(epoch_losses_train))
        print("Load Prev val Loss: ", len(epoch_losses_val))
        print("Load Prev miou Loss: ", len(mertics_val_miou))
        print("Load Prev pixAcc Loss: ", len(mertics_val_pixAcc))


    min_train_loss = np.inf
    min_valid_loss = np.inf
    max_fwiou = -np.inf
    max_pixacc = -np.inf

    tensorboard_path = os.path.join(model_dir, "runs")
    if not os.path.exists(tensorboard_path):
        os.makedirs(tensorboard_path)

    evaluator = Evaluator(seg_cls)