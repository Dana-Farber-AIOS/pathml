from scipy.ndimage import gaussian_filter

import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F

from pathml.preprocessing.wsi import HESlide
from pathml.ml.efficientnet import EfficientNet

from pathml.datasets.deepfocus import DeepFocusDataModule 

def main():
    net = EfficientNetBlur()
    datamodule = DeepFocusDataModule(
            data_dir = "tests/testdata/deepfocus/" 
            batch_size = 8, 
            download = True, 
            transforms = None
    )
    model = EfficientNetBlur().to(device)
    train_loader = datamodule.train_dataloader()
    test_loader = datamodule.test_dataloader()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    # train on deepfocus data
    for epoch in range(1, 100):
        train(epoch)
        test(epoch)
    # finetune on synthetic data mixed with deepfocus data


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

    def loss(preds, targets):
        pass


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        imgs, targets = data.float().to(device)
        optimizer.zero_grad()
        pred = model(imgs)
        loss = model.loss(pred, targets)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        print(f"{Epoch} Average Training Loss: {train_loss/len(train_loader.dataset)}")

def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            imgs, targets = data.float().to(device)
            pred = model(imgs)
            loss = model.loss(pred, targets)
            test_loss += loss.item()
            print(f"{Epoch} Test Loss: {test_loss}")


def finetune(model, dataset, n_augmented = None):
    """
    Fine tune model by mixing held out fold of pretraining set with synthetic blur from dataset of interest
    """
    # generate augmented dataset

    # mix augmented data and held-out fold

    # train model



def syntheticblur(patch, sigma, filtertype='gaussian'):
    """
    Apply synthetic blur to patch.
    Gaussian filter with standard deviation sigma.
    """
    if filtertype == 'gaussian':
        blurred = gaussian_filter(patch, sigma)
    return blurred
