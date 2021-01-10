import os
import ntpath
import h5py
from pathlib import path

import torch.utils.data
from torch.utils.data import Dataset, DataLoader

from pathml.datasets.base import BaseDataModule, BaseDataset
from pathml.ml.utils import download_url

class DeepFocusDataModule(BaseDataModule):
    """
    Data from Cia lab DeepFocus experiments
    https://github.com/cialab/DeepFocus
    """
    def __init__(self, 
            data_dir, 
            download = False,
            shuffle = True,
            transforms = None,
            batch_size=8
    ):
        if download is True:
            self._download_deepfocus(data_dir) 
        else:
            assert data_dir.isdir(), f"download is False but data directory does not exist"

        self.shuffle = shuffle
        self.transforms = trnasforms
        self.batch_size = batch_size

    def __len__(self):
        return len(self.datah5['X'])

    def train_dataloader(self, split = 1):
        return data.DataLoader(
                dataset = self._get_dataset(fold_ix = split),
                batch_size = self.batch_size.
                shuffle = self.shuffle
        )
        
    def valid_dataloader(self, split = 2):
        return data.DataLoader(
                dataset = self._get_dataset(fold_ix = split),
                batch_size = self.batch_size.
                shuffle = self.shuffle
        )

    def test_dataloader(self, split=3):
        return data.DataLoader(
                dataset = self._get_dataset(fold_ix = split),
                batch_size = self.batch_size.
                shuffle = self.shuffle
        )
    
    def _get_dataset(self, fold_idx = None):
        return DeepFocusDataset(
                data_dir = self.data_dir,
                fold_idx = fold_idx,
                transforms = self.transforms
        )

    def _download_deepfocus(self, root):
        if self._check_integrity():
            print('File already downloaded')
            return
        # TODO: add md5 checksum
        download_url('https://zenodo.org/record/1134848/files/outoffocus2017_patches5Classification.h5', root)
        # TODO: clean dataset?

    def _checkintegrity(self) -> bool:
        # TODO: check hash of file
        return os.path.exists(path)

class DeepFocusDataset(BaseDataset):
    def __init__(self,
            data_dir,
            fold_ix=None,
            transforms=None):
        self.datah5 = h5py.File(str(data_dir + 'outoffocus2017_patches5Classification.h5'), 'r')
        # TODO: decide size of folds
        if fold_ix == 1:
            self.datah5 = self.datah5[]
        if fold_ix == 2:
            self.datah5 = self.datah5[]
        if fold_ix == 3:
            self.datah5 = self.datah5[]

    def __len__(self):
        return len(self.datah5['X'])

    def __getitem__(self, index: int):
       img = self.datah5['X'][index]
       target = self.datah5['Y'][index]
       return img, target

