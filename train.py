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

from dataset import MakeTrainValidSet, MapillaryDataset
from model.backbone_model import DeeplabV3
from lr_scheduler import Poly
from utils import add_weight_decay, Evaluator, draw_label, denormalize
from utils_losses import DiceLoss, CE_DiceLoss, CrossEntropyLoss2d, LovaszSoftmax

def PlotNumberPic(values, v_name, filepath, plotpath, xlabel="epoch", ylabel="loss"):
    with open(filepath, "wb") as file:
        pickle.dump(values, file)
    plt.figure(1)
    plt.plot(values, "k^")
    plt.plot(values, "k")
    plt.title(v_name)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.savefig(plotpath)
    plt.close(1)

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


    # optimizer = torch.optim.Adam(params=trainable_params, **paramsOptim)
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

    writer = SummaryWriter(tensorboard_path)

    for epoch in range(init_epoch, num_epochs):
        print ("epoch: %d/%d" % (epoch+1, num_epochs))
        model.train() # (set in training mode, this affects BatchNorm and dropout)
        batch_losses = []
        # assert(0)


        ############################################################################
        # train:
        ############################################################################
        
        for step, (imgs, label_imgs) in enumerate(train_loader):
            

            imgs = imgs.cuda() # (shape: (batch_size, 3, img_h, img_w))
            label_imgs = label_imgs.type(torch.LongTensor).cuda() # (shape: (batch_size, img_h, img_w))
            outputs = model(imgs)

            loss = criterion(outputs, label_imgs)
            loss_value = loss.data.cpu().numpy()
            batch_losses.append(loss_value)

            optimizer.zero_grad() # (reset gradients)
            loss.backward() # (compute gradients)
            optimizer.step() # (perform optimization step)
            break

        lr_scheduler.step()

        epoch_losses_train.append(np.mean(batch_losses))
        TRAINLOSS_str = "train loss: {:.4f}".format(np.mean(batch_losses))
        writer.add_scalar('train/loss_epoch', np.mean(batch_losses), epoch)

        PlotNumberPic(epoch_losses_train,
                    v_name = "train loss per epoch",
                    filepath = "%s/epoch_losses_train.pkl" % model_dir,
                    plotpath = "%s/epoch_losses_train.png" % model_dir,
                    xlabel="epoch",
                    ylabel="loss")

        ############################################################################
        # val:
        ############################################################################

        model.eval() # (set in evaluation mode, this affects BatchNorm and dropout)
        batch_losses = []
        mertics_miou = []
        mertics_pixAcc = []

        for step, (imgs, label_imgs) in enumerate(val_loader):
            with torch.no_grad(): 
                imgs = imgs.cuda() # (shape: (batch_size, 3, img_h, img_w))
                label_imgs = label_imgs.type(torch.LongTensor).cuda() # (shape: (batch_size, img_h, img_w))

                outputs = model(imgs) # (shape: (batch_size, num_classes, img_h, img_w))

                # compute the loss:
                loss = criterion(outputs, label_imgs)
                loss_value = loss.data.cpu().numpy()
                batch_losses.append(loss_value)
                
                outputs = outputs.data.cpu().numpy()
                outputs = np.argmax(outputs, axis=1)
                label_imgs = label_imgs.data.cpu().numpy()
                evaluator.add_batch(label_imgs, outputs)

                if step == 0:
                    grid_image = make_grid(denormalize(imgs[:3].data.cpu()), 3)
                    writer.add_image('Image', grid_image, epoch)
                    
                    grid_image = make_grid(draw_label(label_imgs[:3]), 3)
                    writer.add_image('Label', grid_image, epoch)

                    grid_image = make_grid(draw_label(outputs[:3]), 3)
                    writer.add_image('Prediction', grid_image, epoch)
                break

        Acc = evaluator.Pixel_Accuracy()
        Acc_class = evaluator.Pixel_Accuracy_Class()
        mIoU = evaluator.Mean_Intersection_over_Union()
        FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()

        
        mertics_val_miou.append(mIoU)
        mertics_val_pixAcc.append(Acc)
        mertics_val_FWIoU.append(FWIoU)
        writer.add_scalar('val/Acc_class', Acc_class, epoch)

        
        MIOU_str = "miou: {:.4f}".format(mIoU)
        writer.add_scalar('val/mIoU', mIoU, epoch)
        PlotNumberPic(mertics_val_miou,
                    v_name = "miou per epoch",
                    filepath = "%s/epoch_miou_val.pkl" % model_dir,
                    plotpath = "%s/epoch_miou_val.png" % model_dir,
                    xlabel="epoch",
                    ylabel="miou")

        FWIoU_str = "FWIoU: {:.4f}".format(FWIoU)
        writer.add_scalar('val/fwIoU', FWIoU, epoch)
        PlotNumberPic(mertics_val_miou,
                    v_name = "FWIoU per epoch",
                    filepath = "%s/epoch_FWIoU_val.pkl" % model_dir,
                    plotpath = "%s/epoch_FWIoU_val.png" % model_dir,
                    xlabel="epoch",
                    ylabel="FWIoU")

        PIXACC_str = "pixAcc: {:.4f}".format(Acc)
        writer.add_scalar('val/Acc', Acc, epoch)
        PlotNumberPic(mertics_val_pixAcc,
                    v_name = "pixAcc per epoch",
                    filepath = "%s/epoch_pixAcc_val.pkl" % model_dir,
                    plotpath = "%s/epoch_pixAcc_val.png" % model_dir,
                    xlabel="epoch",
                    ylabel="pixAcc")

        epoch_loss = np.mean(batch_losses)
        epoch_losses_val.append(epoch_loss)
        VALIDLOSS_str = "val loss: {:.3f}".format(epoch_loss)
        writer.add_scalar('val/loss_epoch', epoch_loss, epoch)
        PlotNumberPic(epoch_losses_val,
                    v_name = "val per epoch",
                    filepath = "%s/epoch_losses_val.pkl" % model_dir,
                    plotpath = "%s/epoch_losses_val.png" % model_dir,
                    xlabel="epoch",
                    ylabel="loss")

        ############################################################################
        # save the model weights to disk
        ############################################################################            
        

        print("\t" + TRAINLOSS_str + " , " + VALIDLOSS_str + " , "\
                + MIOU_str + " , " + FWIoU_str + ' , ' + \
                PIXACC_str + " , lr:{:.8f}".format(optimizer.param_groups[0]['lr']))
        
        checkpoint_path = ckpt_path + "/model_" + params.model_id +"_epoch_" + str(epoch+1) + ".pth"
        torch.save({ 'epoch': epoch, 
                    'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                    'lr': optimizer.param_groups[0]['lr']
                    }, ckpt_path + "/last.pth" )
        
        if min_train_loss > epoch_losses_train[-1]:
            min_train_loss = epoch_losses_train[-1]
            torch.save({ 'epoch': epoch, 
                    'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                    'lr': optimizer.param_groups[0]['lr'],
                    "value": min_train_loss
                    }, ckpt_path + "/best_train.pth" )
        
        if min_valid_loss > epoch_losses_val[-1]:
            min_train_loss = epoch_losses_val[-1]
            torch.save({ 'epoch': epoch, 
                    'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                    'lr': optimizer.param_groups[0]['lr'],
                    "value": min_valid_loss
                    }, ckpt_path + "/best_valid.pth" )

        if max_fwiou < FWIoU:
            max_fwiou = FWIoU
            torch.save({ 'epoch': epoch, 
                    'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                    'lr': optimizer.param_groups[0]['lr'],
                    "value": max_fwiou
                    }, ckpt_path + "/best_fwiou.pth" )
        
        if max_pixacc < Acc:
            max_pixacc = Acc
            torch.save({ 'epoch': epoch, 
                    'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                    'lr': optimizer.param_groups[0]['lr'],
                    "value": max_pixacc
                    }, ckpt_path + "/best_pixacc.pth" )


