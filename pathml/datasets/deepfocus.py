"""
Copyright 2021, Dana-Farber Cancer Institute and Weill Cornell Medicine
License: GNU GPL 2.0
"""

import hashlib
import os
from loguru import logger
from pathml.logging.utils import *
from pathlib import Path

import h5py
from pathml.datasets.base_data_module import BaseDataModule
from pathml.utils import download_from_url
from torch.utils.data import DataLoader, Dataset


class DeepFocusDataModule(BaseDataModule):
    """
    DataModule for the DeepFocus dataset. The DeepFocus dataset comprises four slides from different patients,
    each with four different stains (H&E, Ki67, CD21, and CD10) for a total of 16 whole-slide images.
    For each slide, a region of interest (ROI) of approx 6mm^2 was scanned at 40x magnification with
    an Aperio ScanScope on nine different focal planes, generating 216,000 samples with varying amounts
    of blurriness. Tiles with offset values between [-0.5μm, 0.5μm] are labeled as in-focus and
    the rest of the images are labeled as blurry.

    See: https://github.com/cialab/DeepFocus

    Args:
        data_dir (str): file path to directory containing data.
        download (bool, optional): Whether to download the data. If ``True``, checks whether data files exist in
            ``data_dir`` and downloads them to ``data_dir`` if not.
            If ``False``, checks to make sure that data files exist in ``data_dir``. Default ``False``.
        shuffle (bool, optional): Whether to shuffle images. Defaults to ``True``.
        transforms (optional): Data augmentation transforms to apply to images.
        batch_size (int, optional): batch size for dataloaders. Defaults to 8.

    Reference:
        Senaras, C., Niazi, M.K.K., Lozanski, G. and Gurcan, M.N., 2018. DeepFocus: detection of out-of-focus regions in whole slide digital images using deep learning. PloS one, 13(10), p.e0205387.
    """

    def __init__(
        self, data_dir, download=False, shuffle=True, transforms=None, batch_size=8
    ):
        self.data_dir = Path(data_dir)
        if download:
            self._download_deepfocus(self.data_dir)
        else:
            assert (
                self._check_integrity()
            ), f"download is False but data directory does not exist or md5 checksum failed"
        self.shuffle = shuffle
        self.transforms = transforms
        self.batch_size = batch_size

    @property
    def train_dataloader(self):
        return DataLoader(
            dataset=self._get_dataset(fold_ix=1),
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            pin_memory=True,
        )

    @property
    def valid_dataloader(self):
        return DataLoader(
            dataset=self._get_dataset(fold_ix=2),
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            pin_memory=True,
        )

    @property
    def test_dataloader(self):
        return DataLoader(
            dataset=self._get_dataset(fold_ix=3),
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            pin_memory=True,
        )

    def _get_dataset(self, fold_ix=None):
        return DeepFocusDataset(
            data_dir=self.data_dir, fold_ix=fold_ix, transforms=self.transforms
        )

    def _download_deepfocus(self, root):
        if self._check_integrity():
            logger.info("File already downloaded with correct hash.")
            return
        self.data_dir.mkdir(parents=True, exist_ok=True)
        download_from_url(
            "https://zenodo.org/record/1134848/files/outoffocus2017_patches5Classification.h5",
            root,
        )

    def _check_integrity(self) -> bool:
        if os.path.exists(
            self.data_dir / Path("outoffocus2017_patches5Classification.h5")
        ):
            filename = self.data_dir / Path("outoffocus2017_patches5Classification.h5")
            correctmd5 = "ba7b4a652c2a5a7079b216edd267b628"
            with open(filename, "rb") as f:
                fhash = hashlib.md5()
                while chunk := f.read(8192):
                    fhash.update(chunk)
                filemd5 = fhash.hexdigest()
            return correctmd5 == filemd5
        return False


class DeepFocusDataset(Dataset):
    def __init__(self, data_dir, fold_ix=None, transforms=None):
        self.datah5 = h5py.File(
            str(data_dir / "outoffocus2017_patches5Classification.h5"), "r"
        )
        # all
        if fold_ix is None:
            self.X = self.datah5["X"]
            self.Y = self.datah5["Y"]
        # train 80%
        if fold_ix == 1:
            self.X = self.datah5["X"][0:163199]
            self.Y = self.datah5["Y"][0:163199]
        # valid 10%
        if fold_ix == 2:
            self.X = self.datah5["X"][163200:183600]
            self.Y = self.datah5["Y"][163200:183600]
        # test 10%
        if fold_ix == 3:
            self.X = self.datah5["X"][183601:203999]
            self.Y = self.datah5["Y"][183601:203999]

    def __len__(self):
        return len(self.datah5["X"])

    def __getitem__(self, index: int):
        img = self.datah5["X"][index]
        target = self.datah5["Y"][index]
        return img, target
