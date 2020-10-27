import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
sys.path.append("./model")
from resnet import ResNet18_OS16, ResNet34_OS16, ResNet50_OS16, ResNet101_OS16, ResNet152_OS16, ResNet18_OS8, ResNet34_OS8
from aspp import ASPP, ASPP_Bottleneck

class DeeplabV3(nn.Module):
    def __init__(self, model_id, project_dir, num_classes, FeatureExtractor = 0):
        super(DeeplabV3, self).__init__()
        self.num_classes = num_classes

        self.model_id = model_id
        self.project_dir = project_dir
        self.create_dirs()

        # NOTE! specify the type of ResNet here
        # self.resnet = ResNet34_OS16()
        if FeatureExtractor == 0:
            self.resnet = ResNet18_OS16()
        elif FeatureExtractor == 1:
            self.resnet = ResNet34_OS16()
        elif FeatureExtractor == 2:
            self.resnet = ResNet18_OS8()
        elif FeatureExtractor == 3:
            self.resnet = ResNet34_OS8()
        elif FeatureExtractor == 4:
            self.resnet = ResNet50_OS16()
        elif FeatureExtractor == 5:
            self.resnet = ResNet101_OS16()
        elif FeatureExtractor == 6:
            self.resnet = ResNet152_OS16()
        
        
        if FeatureExtractor in [0, 1, 2, 3]:
            self.aspp = ASPP(num_classes=self.num_classes)
        else:
            self.aspp = ASPP_Bottleneck(num_classes=self.num_classes)

    def forward(self, x):
        # (x has shape (batch_size, 3, h, w))

        h = x.size()[2]
        w = x.size()[3]

        feature_map = self.resnet(x)
        # (shape: (batch_size, 512, h/16, w/16)) 
        # (assuming self.resnet is ResNet18_OS16 or ResNet34_OS16. 
        # If self.resnet is ResNet18_OS8 or ResNet34_OS8, it will be (batch_size, 512, h/8, w/8). 
        # If self.resnet is ResNet50-152, it will be (batch_size, 4*512, h/16, w/16))

        output = self.aspp(feature_map) # (shape: (batch_size, num_classes, h/16, w/16))

        output = nn.Upsample(size=(h, w), mode="bilinear", align_corners=True)(output)
        # output = F.upsample(output, size=(h, w), mode="bilinear") 
        # (shape: (batch_size, num_classes, h, w))

        return output

    def create_dirs(self):
        self.logs_dir = self.project_dir + "logs"
        self.model_dir = self.logs_dir + "/{}".format(self.model_id)
        self.checkpoints_dir = self.model_dir + "/checkpoints"

        if not os.path.exists(self.logs_dir):
            os.makedirs(self.logs_dir)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
            os.makedirs(self.checkpoints_dir)