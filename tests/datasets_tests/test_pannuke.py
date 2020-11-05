import pytest
import urllib

from pathml.datasets.pannuke import PanNukeDataModule


"""
Commenting this out because I don't think it makes sense to download the entire dataset just to test it.
But do need to test the PanNukeDataModule class.... need to think of a smarter way to test it though

@pytest.mark.parametrize("batch_size", [8, 16])
@pytest.mark.parametrize("nucleus_type_labels", [True, False])
def test_batches(batch_size, nucleus_type_labels):
    pannuke = PanNukeDataModule(
        data_dir = "data/pannuke",
        download = True,
        batch_size = batch_size,
        nucleus_type_labels = nucleus_type_labels
    )

    train_dataloader = pannuke.train_dataloader
    valid_dataloader = pannuke.valid_dataloader
    test_dataloader = pannuke.test_dataloader

    for dl in [train_dataloader, valid_dataloader, test_dataloader]:
        ims, masks, tissues = next(iter(dl))
        assert ims.shape == (batch_size, 256, 256, 3)
        if nucleus_type_labels:
            assert masks.shape == (batch_size, 256, 256, 6)
        else:
            assert masks.shape == (batch_size, 256, 256)
        assert len(tissues) == batch_size
"""


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