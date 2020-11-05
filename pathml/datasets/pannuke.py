import zipfile
import os
import numpy as np
import torch.utils.data as data
from warnings import warn

from pathml.datasets.utils import download_from_url


class PanNukeDataset(data.Dataset):
    """
    Dataset object for PanNuke dataset

    Tissue types: Breast, Colon, Bile-duct, Esophagus, Uterus, Lung, Cervix, Head&Neck, Skin, Adrenal Gland, Kidney,
    Stomach, Prostate, Testis, Liver, Thyroid, Pancreas, Ovary, Bladder

    masks are arrays of 6 channel instance-wise masks (0:
    Neoplastic cells, 1: Inflammatory, 2: Connective/Soft tissue cells, 3: Dead Cells, 4: Epithelial, 5: Background)
    If ``nucleus_type_labels is False`` then only a single channel mask will be returned, which is the inverse of the
    'Background' mask (i.e. nucleus pixels are 1.). Otherwise, the full 6-channel masks will be returned.

    If using transforms for data augmentation, the transform must accept two arguments (image and mask) and return a
    dict with "image" and "mask" keys.
    See example here https://albumentations.ai/docs/getting_started/mask_augmentation
    """
    def __init__(self, data_dir, fold_ix, transforms=None, classification=False):
        self.data_dir = data_dir
        self.fold_ix = fold_ix
        self.transforms = transforms
        self.classification = classification

        impath = os.path.join(data_dir, f"Fold {self.fold_ix}/images/fold{fold_ix}/images.npy")
        typepath = os.path.join(data_dir, f"Fold {self.fold_ix}/images/fold{fold_ix}/types.npy")
        maskpath = os.path.join(data_dir, f"Fold {self.fold_ix}/masks/fold{fold_ix}/masks.npy")

        self.images = np.load(impath, mmap_mode = 'r+').astype(np.uint8)
        self.types = np.load(typepath, mmap_mode = 'r+')
        self.masks = np.load(maskpath, mmap_mode = 'r+').astype(np.uint8)

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, ix):
        im = self.images[ix, ...]
        tissue_type = self.types[ix]
        if self.classification is False:
            # only look at "background" mask
            mask = self.masks[ix, ..., 5]
            # invert so that ones are nuclei pixels
            mask = 1 - mask
        else:
            mask = self.masks[ix, ...]

        if self.transforms is not None:
            transformed = self.transforms(image = im, mask = mask)
            im = transformed["image"]
            mask = transformed["mask"]

        return im, mask, tissue_type


class PanNukeDataModule:
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
            {1, 2, 3} corresponding to the following splits:

                1. Training: Fold 1; Validation: Fold 2; Testing: Fold 3
                2. Training: Fold 2; Validation: Fold 1; Testing: Fold 3
                3. Training: Fold 3; Validation: Fold 2; Testing: Fold 1

            Defaults to 1.
        batch_size (int, optional): batch size for dataloaders. Defaults to 16.

    References
        Gamper, J., Koohbanani, N.A., Benet, K., Khuram, A. and Rajpoot, N., 2019, April. PanNuke: an open pan-cancer
        histology dataset for nuclei instance segmentation and classification. In European Congress on Digital
        Pathology (pp. 11-19). Springer, Cham.

        Gamper, J., Koohbanani, N.A., Graham, S., Jahanifar, M., Khurram, S.A., Azam, A., Hewitt, K. and Rajpoot, N.,
        2020. PanNuke Dataset Extension, Insights and Baselines. arXiv preprint arXiv:2003.10778.
    """

    def __init__(self, data_dir, download=False, shuffle=True, transforms=None,
                 nucleus_type_labels=False, split=1, batch_size=16):
        self.data_dir = data_dir
        self.download = download
        if download:
            self._download_pannuke(self.data_dir)
        else:
            # check that files exist
            for f in [1, 2, 3]:
                p = os.path.join(data_dir, f"Fold {f}")
                assert os.path.isdir(p), \
                    f"Error: `download is False` but PanNuke data for Fold {f} not found at {p}"
        self.shuffle = shuffle
        self.transforms = transforms
        self.classification = nucleus_type_labels
        assert split in [1, 2, 3], f"Error: input split {split} not valid. Must be one of [1, 2, 3]."
        self.split = split
        self.batch_size = batch_size

    def _get_dataset(self, fold_ix):
        return PanNukeDataset(
            data_dir = self.data_dir,
            fold_ix = fold_ix,
            transforms = self.transforms,
            classification = self.classification
        )

    @staticmethod
    def _download_pannuke(download_dir):
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
                # delete zip files
                os.remove(path = path)
            else:
                warn(f"Skipping download of fold {fold_ix}, using local data found at {p}")

    @property
    def train_dataloader(self):
        """
        Dataloader for training set.
        Yields (image, mask, tissue_type)
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
        Yields (image, mask, tissue_type)
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
        Yields (image, mask, tissue_type)
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
