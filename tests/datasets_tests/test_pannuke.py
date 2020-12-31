import pytest
import urllib
import numpy as np
import cv2

from pathml.datasets.pannuke import PanNukeDataModule, PanNukeDataset


def create_fake_pannuke_data(target_dir, n_fold=16):
    """
    create some fake images and masks in target_dir

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






# TODO add tests for dataloaders
# TODO add tests for _process_downloaded_pannuke(), _download_pannuke(), and _clean_up_download_pannuke()


def check_pannuke_data_urls():
    # make sure that the urls for the pannuke data are still valid!
    for fold_ix in [1, 2, 3]:
        url = f"https://warwick.ac.uk/fac/sci/dcs/research/tia/data/pannuke/fold_{fold_ix}.zip"
        r = urllib.request.urlopen(url)
        # HTTP status code 200 means "OK"
        assert r.getcode() == 200


def check_wrong_path_download_false():
    with pytest.raises(AssertionError):
        pannuke = PanNukeDataModule(data_dir = "wrong/path/to/pannuke", download = False)