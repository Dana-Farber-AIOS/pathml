import os
import ntpath
import h5py
from pathlib import Path
import hashlib

import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader

from pathml.datasets.base import BaseDataModule, BaseDataset
from pathml.datasets.utils import download_from_url

class DeepFocusDataModule(BaseDataModule):
    """
    Pytorch DataModule for Cia lab DeepFocus dataset.
    https://github.com/cialab/DeepFocus
    """
    def __init__(self, 
            data_dir, 
            download = False,
            shuffle = True,
            transforms = None,
            batch_size=8
    ):
        self.data_dir = Path(data_dir)
        if download:
            self._download_deepfocus(self.data_dir) 
        else:
            assert self._check_integrity(), f"download is False but data directory does not exist or md5 checksum failed"
        self.shuffle = shuffle
        self.transforms = transforms
        self.batch_size = batch_size

    @property
    def train_dataloader(self):
        return data.DataLoader(
                dataset = self._get_dataset(fold_ix = 1),
                batch_size = self.batch_size,
                shuffle = self.shuffle,
                pin_memory = True
        )
    
    @property
    def valid_dataloader(self):
        return data.DataLoader(
                dataset = self._get_dataset(fold_ix = 2),
                batch_size = self.batch_size,
                shuffle = self.shuffle,
                pin_memory = True
        )

    @property
    def test_dataloader(self):
        return data.DataLoader(
                dataset = self._get_dataset(fold_ix = 3),
                batch_size = self.batch_size,
                shuffle = self.shuffle,
                pin_memory = True
        )
    
    def _get_dataset(self, fold_ix = None):
        return DeepFocusDataset(
                data_dir = self.data_dir,
                fold_ix = fold_ix,
                transforms = self.transforms
        )

    def _download_deepfocus(self, root):
        if self._check_integrity():
            print('File already downloaded with correct hash.')
            return
        self.data_dir.mkdir(parents=True, exist_ok=True)
        download_from_url('https://zenodo.org/record/1134848/files/outoffocus2017_patches5Classification.h5', root)

    def _check_integrity(self) -> bool:
        if os.path.exists(self.data_dir /  Path('outoffocus2017_patches5Classification.h5')):
            filename = self.data_dir / Path('outoffocus2017_patches5Classification.h5')
            correctmd5 = 'ba7b4a652c2a5a7079b216edd267b628' 
            with open(filename, "rb") as f:
                fhash = hashlib.md5()
                while chunk := f.read(8192):
                    fhash.update(chunk)
                filemd5 = fhash.hexdigest()
            return correctmd5 == filemd5 
        return False
        

class DeepFocusDataset(BaseDataset):
    def __init__(self,
            data_dir,
            fold_ix=None,
            transforms=None):
        self.datah5 = h5py.File(str(data_dir / Path('outoffocus2017_patches5Classification.h5')), 'r')
        # all
        if fold_ix == None:
            self.X = self.datah5['X']
            self.Y = self.datah5['Y']
        # train 80%
        if fold_ix == 1:
            self.X = self.datah5['X'][0:163199]
            self.Y = self.datah5['Y'][0:163199]
        # valid 10%
        if fold_ix == 2:
            self.X = self.datah5['X'][163200:183600]
            self.Y = self.datah5['Y'][163200:183600]
        # test 10%
        if fold_ix == 3:
            self.X = self.datah5['X'][183601:203999]
            self.Y = self.datah5['Y'][183601:203999]

    def __len__(self):
        return len(self.datah5['X'])

    def __getitem__(self, index: int):
       img = self.datah5['X'][index]
       target = self.datah5['Y'][index]
       return img, target
