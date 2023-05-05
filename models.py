import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.feature_extraction import create_feature_extractor
import torch.nn.functional as F
from torch import optim
import copy
import numpy as np

# ----------------- Feature Extraction Models ---------------

class FeResNet50(nn.Module): # ResNet50
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        self.model = models.resnet50(pretrained=pretrained)
        self.freeze_layer()
        fc_input_dim = self.model.fc.in_features
        self.model.fc = nn.Linear(fc_input_dim, num_classes)

    def get_features(self, x):

        # followed https://pytorch.org/vision/stable/feature_extraction.html to get the node ids

        return_nodes = {'avgpool': 'avgpool'}
        fe = create_feature_extractor(self.model, return_nodes=return_nodes)
        feat = fe(x)
        feat = feat['avgpool']
        feat = torch.squeeze(feat)

        return feat # returns feat with shape [batch_size, 2048]

    def freeze_layer(self):
        ct = 0
        for child in self.model.children():
            ct += 1
            if ct < 8:
                for parma in child.parameters():
                    parma.requires_grad = False

    def forward(self, x):
        out = self.model(x)
        return out

# try other feature image models such as EfficientNets, MobileNets, Other ResNets etc

# ----------------- Temporal Models --------------------------

class MultiStageModel(nn.Module):
    def __init__(self, num_stages, num_layers, num_f_maps, dim, num_classes):
        super(MultiStageModel, self).__init__()
        self.stage1 = SingleStageModel(num_layers, num_f_maps, dim, num_classes)
        self.stages = nn.ModuleList([copy.deepcopy(SingleStageModel(num_layers, num_f_maps, num_classes, num_classes)) for s in range(num_stages-1)])

    def forward(self, x):
        out = self.stage1(x)
        outputs = out.unsqueeze(0)
        for s in self.stages:
            out = s(F.softmax(out, dim=1))
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
        return outputs


class SingleStageModel(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super(SingleStageModel, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList([copy.deepcopy(DilatedResidualLayer(2 ** i, num_f_maps, num_f_maps)) for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x):

        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out)
        out = self.conv_out(out)
        return out


class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(DilatedResidualLayer, self).__init__()
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return x + out


