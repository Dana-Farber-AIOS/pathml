import zipfile

import pytest
import urllib
import numpy as np
import cv2

from pathml.datasets.pannuke import PanNukeDataModule, PanNukeDataset, pannuke_multiclass_mask_to_nucleus_mask


def create_fake_pannuke_data(target_dir, n_fold=16):
    """
    create some fake images and masks in target_dir/images and target_dir/masks

    Args:
        target_dir (pathlib.Path): directory where to save the images and masks. 'images' and 'masks' subdirectories
            will be created here.
        n_fold (int): number of images and masks per fold
    """
    folds = [1, 2, 3]
    tissue_types = ["breast", "colon", "head-neck"]

    imdir = target_dir / "images"
    maskdir = target_dir / "masks"

    imdir.mkdir()
    maskdir.mkdir()

    for fold_ix in folds:
        for i in range(n_fold):
            im = np.random.randint(low = 2, high = 254, size = (256, 256, 3), dtype = np.uint8)
            mask = np.random.randint(low = 0, high = 10, size = (256, 256, 6), dtype = np.uint8)
            tissue_type = np.random.choice(tissue_types)

            im_fname = imdir / f"fold{fold_ix}_{i}_{tissue_type}.png"
            im_fname = str(im_fname.resolve())
            mask_fname = maskdir / f"fold{fold_ix}_{i}_{tissue_type}.npy"
            mask_fname = str(mask_fname.resolve())

            cv2.imwrite(im_fname, im)
            np.save(mask_fname, mask)


@pytest.mark.parametrize("fold", [1, 2, 3, None])
@pytest.mark.parametrize("nucleus_type_labels", [True, False])
def test_pannuke_dataset_sizes(tmp_path, fold, nucleus_type_labels):
    n_fold = 16
    create_fake_pannuke_data(tmp_path, n_fold = n_fold)

    pannuke_dataset = PanNukeDataset(data_dir = tmp_path, fold_ix = fold, nucleus_type_labels = nucleus_type_labels)

    # check size of dataset
    if fold in [1, 2, 3]:
        assert len(pannuke_dataset) == n_fold
    else:
        assert len(pannuke_dataset) == 3*n_fold

    # check shapes of individual elements
    im, mask, lab = pannuke_dataset[0]
    assert im.shape == (3, 256, 256)

    if nucleus_type_labels:
        assert mask.shape == (6, 256, 256)
    else:
        assert mask.shape == (256, 256)



def create_fake_pannuke_data_raw(target_dir, fold_size=16):
    """
    Create some fake raw data that mimics file structure of what is downloaded from PanNuke website:
    https://warwick.ac.uk/fac/sci/dcs/research/tia/data/pannuke/

    Args:
        target_dir (pathlib.Path): directory where to save the data.
        fold_size (int): number of images and masks per fold
    """
    folds = [1, 2, 3]
    tissue_types = ["breast", "colon", "head-neck"]

    for fold_ix in folds:
        # create the directories
        images_dir = target_dir / f"Fold {fold_ix}" / "images" / f"fold{fold_ix}"
        masks_dir = target_dir / f"Fold {fold_ix}" / "masks" / f"fold{fold_ix}"
        images_dir.mkdir(parents = True)
        masks_dir.mkdir(parents = True)

        # create the fake data
        types_fold = np.random.choice(tissue_types, size = fold_size)
        masks = np.random.randint(low = 0, high = 10, size = (fold_size, 256, 256, 6))
        ims = np.random.randint(low = 0, high = 254, size = (fold_size, 256, 256, 3))

        # write the data
        np.save(file = str(images_dir / "images.npy"), arr = ims)
        np.save(file = str(images_dir / "types.npy"), arr = types_fold)
        np.save(file = str(masks_dir / "masks.npy"), arr = masks)


def test_process_downloaded_pannuke(tmp_path):
    """ Test the post-processing of the pannuke raw data """
    # make fake data
    fold_size = 16
    create_fake_pannuke_data_raw(tmp_path, fold_size = fold_size)

    # process the fake data
    PanNukeDataModule._process_downloaded_pannuke(tmp_path)

    # check everything
    imdir = tmp_path / "images"
    maskdir = tmp_path / "masks"

    assert imdir.is_dir()
    assert maskdir.is_dir()

    assert len(list(imdir.glob("*"))) == 3*fold_size
    assert len(list(maskdir.glob("*"))) == 3 * fold_size

    for fold_ix in [1, 2, 3]:
        assert len(list(imdir.glob(f"fold{fold_ix}*"))) == fold_size
        assert len(list(maskdir.glob(f"fold{fold_ix}*"))) == fold_size


@pytest.mark.parametrize("hovernet_preprocess", [True, False])
@pytest.mark.parametrize("split", [1, 2, 3, None])
@pytest.mark.parametrize("nucleus_type_labels", [True, False])
def test_pannuke_datamodule(tmp_path, split, nucleus_type_labels, hovernet_preprocess):
    # make fake data
    fold_size = 16
    create_fake_pannuke_data(tmp_path, n_fold = fold_size)

    batch_size = 8
    pannuke = PanNukeDataModule(data_dir = tmp_path, nucleus_type_labels = nucleus_type_labels,
                                split = split, download = False, transforms = None,
                                batch_size = batch_size, hovernet_preprocess = hovernet_preprocess)

    train = pannuke.train_dataloader
    valid = pannuke.valid_dataloader
    test = pannuke.test_dataloader

    for loader in [train, test, valid]:
        # make sure everything is correct dimensions
        if hovernet_preprocess:
            im, mask, hv, tissue_types = next(iter(loader))
            assert hv.shape == (batch_size, 2, 256, 256)
        else:
            im, mask, tissue_types = next(iter(loader))

        assert im.shape == (batch_size, 3, 256, 256)
        if nucleus_type_labels:
            assert mask.shape == (batch_size, 6, 256, 256)
        else:
            assert mask.shape == (batch_size, 256, 256)

        assert len(tissue_types) == batch_size and all([isinstance(t, str) for t in tissue_types])


def test_clean_up_download_pannuke(tmp_path):
    # first create the files and dirs to delete
    for fold_ix in [1, 2, 3]:
        with zipfile.ZipFile(tmp_path / f"fold_{fold_ix}.zip", 'w') as myzip:
            myzip.writestr('fake_pannuke_data.txt', "NYE 2020 - happy new year!")
        downloaded_dir = tmp_path / f"Fold {fold_ix}"
        downloaded_dir.mkdir()

    # now call cleanup
    PanNukeDataModule._clean_up_download_pannuke(tmp_path)

    # now make sure that the files/dirs were deleted
    for fold_ix in [1, 2, 3]:
        zfile = tmp_path / f"fold_{fold_ix}.zip"
        downloaded_dir = tmp_path / f"Fold {fold_ix}"
        assert not zfile.exists()
        assert not downloaded_dir.exists()


def check_pannuke_data_urls():
    # make sure that the urls for the pannuke data are still valid!
    for fold_ix in [1, 2, 3]:
        url = f"https://warwick.ac.uk/fac/sci/dcs/research/tia/data/pannuke/fold_{fold_ix}.zip"
        r = urllib.request.urlopen(url)
        # HTTP status code 200 means "OK"
        assert r.getcode() == 200


def check_wrong_path_download_false_fails():
    with pytest.raises(AssertionError):
        pannuke = PanNukeDataModule(data_dir = "wrong/path/to/pannuke", download = False)


def test_pannuke_multiclass_mask_to_nucleus_mask():
    mask = np.random.randint(low = 0, high = 10, size = (6, 256, 256), dtype = np.uint8)
    mask_1c = pannuke_multiclass_mask_to_nucleus_mask(mask)
    assert mask_1c.shape == (256, 256)

# TODO add tests for _download_pannuke()