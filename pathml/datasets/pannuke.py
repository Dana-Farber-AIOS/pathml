import zipfile
import os
import numpy as np
import torch
import torch.utils.data as data
from warnings import warn
from pathlib import Path
import re
import cv2
import shutil

from pathml.datasets.base import BaseTileDataset, BaseDataModule
from pathml.datasets.utils import download_from_url
from pathml.ml.hovernet import compute_hv_map


class PanNukeDataset(BaseTileDataset):
    """
    Dataset object for PanNuke dataset

    Tissue types: Breast, Colon, Bile-duct, Esophagus, Uterus, Lung, Cervix, Head&Neck, Skin, Adrenal-Gland, Kidney,
    Stomach, Prostate, Testis, Liver, Thyroid, Pancreas, Ovary, Bladder

    masks are arrays of 6 channel instance-wise masks
    (0: Neoplastic cells, 1: Inflammatory, 2: Connective/Soft tissue cells, 3: Dead Cells, 4: Epithelial, 5: Background)
    If ``classification is False`` then only a single channel mask will be returned, which is the inverse of the
    'Background' mask (i.e. nucleus pixels are 1.). Otherwise, the full 6-channel masks will be returned.

    If using transforms for data augmentation, the transform must accept two arguments (image and mask) and return a
    dict with "image" and "mask" keys.
    See example here https://albumentations.ai/docs/getting_started/mask_augmentation

    Args:
        data_dir: Path to PanNuke data. Should contain an 'images' directory and a 'masks' directory.
            Images should be 256x256 RGB in a format that can be read by `cv2.imread()` (e.g. png).
            Masks should be .npy files of shape (6, 256, 256).
        fold_ix: Index of which fold of PanNuke data to use. One of 1, 2, or 3. If ``None``, ignores the folds and uses
            the entire PanNuke dataset. Defaults to ``None``.
        transforms: Transforms to use for data augmentation. Must accept two arguments (image and mask) and return a
            dict with "image" and "mask" keys. If ``None``, no transforms are applied. Defaults to ``None``.
        nucleus_type_labels (bool, optional): Whether to provide nucleus type labels, or binary nucleus labels.
            If ``True``, then masks will be returned with six channels, corresponding to

                0. Neoplastic cells
                1. Inflammatory
                2. Connective/Soft tissue cells
                3. Dead Cells
                4. Epithelial
                5. Background

            If ``False``, then the returned mask will have a single channel, with zeros for background pixels and ones
            for nucleus pixels (i.e. the inverse of the Background mask). Defaults to ``False``.
        hovernet_preprocess (bool): Whether to perform preprocessing specific to HoVer-Net architecture. If ``True``,
            the center of mass of each nucleus will be computed, and an additional mask will be returned with the
            distance of each nuclear pixel to its center of mass in the horizontal and vertical dimensions.
            This corresponds to Gamma(I) from the HoVer-Net paper. Defaults to ``False``.
    """
    def __init__(self, data_dir, fold_ix=None, transforms=None, nucleus_type_labels=False, hovernet_preprocess=False):
        self.data_dir = data_dir
        self.fold_ix = fold_ix
        self.transforms = transforms
        self.nucleus_type_labels = nucleus_type_labels
        self.hovernet_preprocess = hovernet_preprocess

        data_dir = Path(data_dir)

        # dirs for images, masks
        imdir = data_dir / "images"
        maskdir = data_dir / "masks"

        # stop if the images and masks directories don't already exist
        assert imdir.is_dir(), f"Error: 'images' directory not found: {imdir}"
        assert maskdir.is_dir(), f"Error: 'masks' directory not found: {maskdir}"

        if self.fold_ix is None:
            self.impaths = list(imdir.glob("*"))
            self.maskpaths = list(maskdir.glob("*"))
        else:
            self.impaths = list(imdir.glob(f"fold{fold_ix}*"))
            self.maskpaths = list(maskdir.glob(f"fold{fold_ix}*"))

    def __len__(self):
        return len(self.impaths)

    def __getitem__(self, ix):
        impath = self.impaths[ix]
        maskpath = self.maskpaths[ix]
        tissue_type = str(impath.stem).split(sep = "_")[2]

        im = cv2.imread(str(impath))
        mask = np.load(str(maskpath))

        if self.nucleus_type_labels is False:
            # only look at "background" mask in last channel
            mask = mask[5, :, :]
            # invert so that ones are nuclei pixels
            mask = 1 - mask

        if self.transforms is not None:
            transformed = self.transforms(image = im, mask = mask)
            im = transformed["image"]
            mask = transformed["mask"]

        # swap channel dim to pytorch standard (C, H, W)
        im = im.transpose((2, 0, 1))

        # compute hv map
        if self.hovernet_preprocess:
            if self.nucleus_type_labels:
                # sum across mask channels to squash mask channel dim to size 1
                # don't sum the last channel, which is background!
                mask_1c = pannuke_multiclass_mask_to_nucleus_mask(mask)
            else:
                mask_1c = mask
            hv_map = compute_hv_map(mask_1c)

        if self.hovernet_preprocess:
            out = torch.from_numpy(im), torch.from_numpy(mask), torch.from_numpy(hv_map), tissue_type
        else:
            out = torch.from_numpy(im), torch.from_numpy(mask), tissue_type

        return out


def pannuke_multiclass_mask_to_nucleus_mask(multiclass_mask):
    """
    Convert multiclass mask from PanNuke to a single channel nucleus mask.
    Assumes each pixel is assigned to one and only one class. Sums across channels, except the last mask channel
    which indicates background pixels in PanNuke.
    Operates on a single mask.

    Args:
        multiclass_mask (torch.Tensor): Mask from PanNuke, in classification setting. (i.e. ``nucleus_type_labels=True``).
            Tensor of shape (6, 256, 256).

    Returns:
        Tensor of shape (256, 256).
    """
    # verify shape of input
    assert multiclass_mask.ndim == 3 and multiclass_mask.shape[0] == 6, \
        f"Expecting a mask with dims (6, 256, 256). Got input of shape {multiclass_mask.shape}"
    assert multiclass_mask.shape[1] == 256 and multiclass_mask.shape[2] == 256, \
        f"Expecting a mask with dims (6, 256, 256). Got input of shape {multiclass_mask.shape}"
    # ignore last channel
    out = np.sum(multiclass_mask[:-1, :, :], axis = 0)
    return out


class PanNukeDataModule(BaseDataModule):
    """
    DataModule for the PanNuke Dataset. Contains 256px image patches from 19 tissue types with annotations for 5
    nucleus types. For more information, see: https://warwick.ac.uk/fac/sci/dcs/research/tia/data/pannuke

    Args:
        data_dir (str): Path to directory where PanNuke data is
        download (bool, optional): Whether to download the data. If ``True``, checks whether data files exist in
            ``data_dir`` and downloads them to ``data_dir`` if not.
            If ``False``, checks to make sure that data files exist in ``data_dir``. Default ``False``.
        shuffle (bool, optional): Whether to shuffle images. Defaults to ``True``.
        transforms (optional): Data augmentation transforms to apply to images. Transform must accept two arguments:
            (mask and image) and return a dict with "image" and "mask" keys. See an example here:
            https://albumentations.ai/docs/getting_started/mask_augmentation/
        nucleus_type_labels (bool, optional): Whether to provide nucleus type labels, or binary nucleus labels.
            If ``True``, then masks will be returned with six channels, corresponding to

                0. Neoplastic cells
                1. Inflammatory
                2. Connective/Soft tissue cells
                3. Dead Cells
                4. Epithelial
                5. Background

            If ``False``, then the returned mask will have a single channel, with zeros for background pixels and ones
            for nucleus pixels (i.e. the inverse of the Background mask). Defaults to ``False``.
        split (int, optional): How to divide the three folds into train, test, and validation splits. Must be one of
            {1, 2, 3, None} corresponding to the following splits:

                1. Training: Fold 1; Validation: Fold 2; Testing: Fold 3
                2. Training: Fold 2; Validation: Fold 1; Testing: Fold 3
                3. Training: Fold 3; Validation: Fold 2; Testing: Fold 1

            If ``None``, then the entire PanNuke dataset will be used. Defaults to ``None``.
        batch_size (int, optional): batch size for dataloaders. Defaults to 8.
        hovernet_preprocess (bool): Whether to perform preprocessing specific to HoVer-Net architecture. If ``True``,
            the center of mass of each nucleus will be computed, and an additional mask will be returned with the
            distance of each nuclear pixel to its center of mass in the horizontal and vertical dimensions.
            This corresponds to Gamma(I) from the HoVer-Net paper. Defaults to ``False``.

    References
        Gamper, J., Koohbanani, N.A., Benet, K., Khuram, A. and Rajpoot, N., 2019, April. PanNuke: an open pan-cancer
        histology dataset for nuclei instance segmentation and classification. In European Congress on Digital
        Pathology (pp. 11-19). Springer, Cham.

        Gamper, J., Koohbanani, N.A., Graham, S., Jahanifar, M., Khurram, S.A., Azam, A., Hewitt, K. and Rajpoot, N.,
        2020. PanNuke Dataset Extension, Insights and Baselines. arXiv preprint arXiv:2003.10778.
    """

    def __init__(self, data_dir, download=False, shuffle=True, transforms=None,
                 nucleus_type_labels=False, split=None, batch_size=8, hovernet_preprocess=False):
        self.data_dir = Path(data_dir)
        self.download = download
        if download:
            self._download_pannuke(self.data_dir)
        else:
            # make sure that subdirectories exist
            imdir = self.data_dir / "images"
            maskdir = self.data_dir / "masks"
            assert imdir.is_dir(), f"`download is False` but 'images' subdirectory not found at {imdir}"
            assert maskdir.is_dir(), f"`download is False` but 'masks' subdirectory not found at {maskdir}"

        self.shuffle = shuffle
        self.transforms = transforms
        self.nucleus_type_labels = nucleus_type_labels
        assert split in [1, 2, 3, None], f"Error: input split {split} not valid. Must be one of [1, 2, 3] or None."
        self.split = split
        self.batch_size = batch_size
        self.hovernet_preprocess = hovernet_preprocess

    def _get_dataset(self, fold_ix):
        return PanNukeDataset(
            data_dir = self.data_dir,
            fold_ix = fold_ix,
            transforms = self.transforms,
            nucleus_type_labels = self.nucleus_type_labels,
            hovernet_preprocess = self.hovernet_preprocess
        )

    def _download_pannuke(self, download_dir):
        """download PanNuke dataset"""
        for fold_ix in [1, 2, 3]:
            p = os.path.join(download_dir, "Fold " + str(fold_ix))
            # don't download if the directory already exists
            if not os.path.isdir(p):
                print(f"Downloading fold {fold_ix}")
                url = f"https://warwick.ac.uk/fac/sci/dcs/research/tia/data/pannuke/fold_{fold_ix}.zip"
                name = os.path.basename(url)
                download_from_url(url = url, download_dir = download_dir, name = name)
                path = os.path.join(download_dir, name)
                # unzip
                with zipfile.ZipFile(path, 'r') as zip_ref:
                    zip_ref.extractall(download_dir)
            else:
                warn(f"Skipping download of fold {fold_ix}, using local data found at {p}")

        self._process_downloaded_pannuke(download_dir)
        self._clean_up_download_pannuke(download_dir)

    @staticmethod
    def _process_downloaded_pannuke(pannuke_dir):
        """
        Process downloaded .npy files, save individual images & masks.
        That way we can load a single image efficiently without holding entire dataset in memory.
        This must be run after _download_pannuke!
        """
        pannuke_dir = Path(pannuke_dir)

        # dirs for images, masks
        imdir = pannuke_dir / "images"
        maskdir = pannuke_dir / "masks"

        # stop if the output files already exist
        assert not imdir.is_dir(), f"Error: 'images' directory already exists: {imdir}"
        assert not maskdir.is_dir(), f"Error: 'masks' directory already exists: {maskdir}"

        imdir.mkdir()
        maskdir.mkdir()

        for fold_ix in [1, 2, 3]:
            ims_fold_path = pannuke_dir / f"Fold {fold_ix}" / "images" / f"fold{fold_ix}" / "images.npy"
            masks_fold_path = pannuke_dir / f"Fold {fold_ix}" / "masks" / f"fold{fold_ix}" / "masks.npy"
            types_fold_path = pannuke_dir / f"Fold {fold_ix}" / "images" / f"fold{fold_ix}" / "types.npy"

            # make sure the input files exist
            assert ims_fold_path.is_file(), f"Error: image file not found at {ims_fold_path}"
            assert masks_fold_path.is_file(), f"Error: masks file not found at {masks_fold_path}"
            assert types_fold_path.is_file(), f"Error: types file not found at {types_fold_path}"

            ims_fold = np.load(ims_fold_path, mmap_mode = 'r')
            masks_fold = np.load(masks_fold_path, mmap_mode = 'r')
            types_fold = np.load(types_fold_path, mmap_mode = 'r')

            # change masks dims from (B, H, W, C) to (B, C, H, W)
            masks_fold = np.moveaxis(masks_fold, 3, 1)

            fold_size = len(types_fold)

            for j in range(fold_size):
                im = ims_fold[j, ...]
                mask = masks_fold[j, ...]
                tissue_type = types_fold[j]
                # change underscores in tissue type label to dashes
                tissue_type = re.sub(pattern = "_", repl = "-", string = tissue_type)

                im_fname = imdir / f"fold{fold_ix}_{j}_{tissue_type}.png"
                im_fname = str(im_fname.resolve())
                mask_fname = maskdir / f"fold{fold_ix}_{j}_{tissue_type}.npy"
                mask_fname = str(mask_fname.resolve())

                cv2.imwrite(im_fname, im)
                np.save(mask_fname, mask)

    @staticmethod
    def _clean_up_download_pannuke(pannuke_dir):
        """remove files after downloading, unzipping, and processing"""
        p = Path(pannuke_dir)

        for fold_ix in [1, 2, 3]:
            zip_file = p / f"fold_{fold_ix}.zip"
            downloaded_dir = p / f"Fold {fold_ix}"
            zip_file.unlink()
            shutil.rmtree(downloaded_dir)


    @property
    def train_dataloader(self):
        """
        Dataloader for training set.
        Yields (image, mask, tissue_type), or (image, mask, hv, tissue_type) for HoVer-Net
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
        Yields (image, mask, tissue_type), or (image, mask, hv, tissue_type) for HoVer-Net
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
        Yields (image, mask, tissue_type), or (image, mask, hv, tissue_type) for HoVer-Net
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
