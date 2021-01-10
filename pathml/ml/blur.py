from scipy.ndimage import gaussian_filter

import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F

from pathml.preprocessing.wsi import HESlide
from pathml.ml.efficientnet import EfficientNet

class EfficientNetBlur(nn.Module):
    '''
    Frame blur detection as regression task.
    Efficientnet pretrained on tiles with different levels of blur.
    Pretrain on fold 1 of __data__.
    Fine tune on heldout fold of mixed data and synthetic blur from DFCI/HCC.
    '''
    def __init__(self, pretrained=True):
        super(EfficientNetBlur, self).__init__()
        if pretrained is True:
            self.en = EfficientNet.from_pretrained('efficientnet-b0')

    def forward(self, x):
        return self.en(x)

class ResNetBlur(nn.Module):
    def __init__(self):
        super(ResNetBlur, self).__init__()
        pass

    def forward(self, x):
        pass

    def infer(self, x):
        pass

def pretrain(model, dataset):
    """
    Pretrain model on DeepFocus data
    """

def finetune(model, dataset, n_augmented = None):
    """
    fine tune model by mixing held out fold of pretraining set with synthetic blur from dataset of interest
    """
    # generate augmented dataset

    # mix augmented data and held-out fold

    # train model


def train(model, dataset):
    pass


def syntheticblur(patch, sigma, filtertype='gaussian'):
    """
    Apply synthetic blur to patch.
    Gaussian filter with standard deviation sigma.
    """
    if filtertype == 'gaussian':
        blurred = gaussian_filter(patch, sigma)
    return blurred
