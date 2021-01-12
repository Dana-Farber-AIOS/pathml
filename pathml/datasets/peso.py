import zipfile
import os
import numpy as np
import torch
import torch.utils.data as data
from warnings import warn
from pathlib import Path
import cv2

from pathml.datasets.base import BaseSlideDataset, BaseDataModule
from pathml.datasets.utils import download_from_url

class PesoDataModule(BaseDataModule):
    def __init__(self,
            data_dir,
            download=False,
            shuffle=True,
            transforms=None,
            split=None,
            batch_size=8):
        self.data_dir = Path(data_dir)
        self.transforms = transforms
        self.download = download
        if download:
            self._download_peso(self.data_dir)

    def _get_dataset(self, fold_ix=None):
        return PesoDataset(
                data_dir = self.data_dir,
                fold_ix = fold_ix,
                transforms = self.transforms
        )

    def __repr__(self):
        return f"repr=(DataModule for PESO segmentation dataset)"

    def _download_peso(self, download_dir):
        if not os.path.isdir(download_dir):
            print("Downloading Peso Dataset")
            files = ['peso_testset_mapping.csv','peso_testset_png.zip','peso_testset_png_padded.zip','peso_testset_regions.zip','peso_testset_wsi_1.zip','peso_testset_wsi_2.zip','peso_testset_wsi_3.zip','peso_testset_wsi_4.zip','peso_training_colordeconvolution.zip','peso_training_masks.zip','peso_training_masks_corrected.zip','peso_training_wsi_1.zip','peso_training_wsi_2.zip','peso_training_wsi_3.zip','peso_training_wsi_4.zip','peso_training_wsi_5.zip','peso_training_wsi_6.zip']
            url = f'https://zenodo.org/record/1485967/files/'
            for file in files:
                print(f"downloading {file}")
                download_from_url(f"{url}{file}", download_dir) 
                if zipfile.is_zipfile(file):
                    with zipfile.ZipFile(f"{download_dir}/{file}",'r') as zip_ref:
                        zip_ref.extractall(f"{download_dir}/{file}")
        else:
            warn(f'download_dir exists, download canceled')

    @property
    def train_dataloader(self):
        """
        Dataloader for training set.
        """
        return data.DataLoader(
            dataset = self._get_dataset(fold_ix = self.split),
            batch_size = self.batch_size,
            shuffle = self.shuffle
        )

    @property
    def valid_dataloader(self):
        """
        Dataloader for validation set.
        """
        if self.split in [1, 3]:
            fold_ix = 2
        else:
            fold_ix = 1
        return data.DataLoader(
            self._get_dataset(fold_ix = fold_ix),
            batch_size = self.batch_size,
            shuffle = self.shuffle
        )

    @property
    def test_dataloader(self):
        """
        Dataloader for test set.
        """
        if self.split in [1, 2]:
            fold_ix = 3
        else:
            fold_ix = 1
        return data.DataLoader(
            self._get_dataset(fold_ix = fold_ix),
            batch_size = self.batch_size,
            shuffle = self.shuffle
        )

class PesoDataset(BaseSlideDataset):
    """
    Dataset object for Peso dataset.
    
    training data:
    training masks (n=62) with labels __
    training masks corrected (n=25) with manual annotations
    training masks color deconvolution (n=62) with p63, ck8/18 stainings
    wsis (n=62) wsis at 0.48 \mu/pix

    testing data:
    testset regions collection of xml files with outlines of test regions
    testset png 2500x2500 pixel test regions
    testset png padded 3500x3500 pixel regions 500pixel padding
    testset mapping csv file maps test set (1-160) to xml files, benign/cancer labels
    testset wsi (n=40) at 0.48 \mu/pix
    """

    def __init__(self,
            data_dir,
            fold_ix = None,
            transforms = None):
        self.data_dir = data_dir
        self.fold_ix = fold_ix
        self.transforms = transforms

        data_dir = Path(data_dir)

        assert data_dir.isdir(), f"Error: data not found at {data_dir}"

        if not any(fname.endswith('.h5') for fname in os.listdir(self.data_dir)):
            self._makeh5()
        self.data = data_dir / 'peso.h5' 

    def __len__(self):
        pass

    def __getitem__(self, ix):
        pass

    def _makeh5(self):
        pass
        


