import pytest
import requests

from pathml.datasets.pannuke import PanNukeDataModule


"""
Commenting this out because I don't think it makes sense to download the entire dataset just to test it.
But do need to test the PanNukeDataModule class.... need to think of a smarter way to test it though

@pytest.mark.parametrize("batch_size", [8, 16])
@pytest.mark.parametrize("classification", [True, False])
def test_batches(batch_size, classification):
    pannuke = PanNukeDataModule(
        data_dir = "data/pannuke",
        download = True,
        batch_size = batch_size,
        classification = classification
    )

    train_dataloader = pannuke.train_dataloader
    valid_dataloader = pannuke.valid_dataloader
    test_dataloader = pannuke.test_dataloader

    for dl in [train_dataloader, valid_dataloader, test_dataloader]:
        ims, masks, tissues = next(iter(dl))
        assert ims.shape == (batch_size, 256, 256, 3)
        if classification:
            assert masks.shape == (batch_size, 256, 256, 6)
        else:
            assert masks.shape == (batch_size, 256, 256)
        assert len(tissues) == batch_size
"""


def check_pannuke_data_urls():
    # make sure that the urls for the pannuke data are still valid!
    for fold_ix in [1, 2, 3]:
        url = f"https://warwick.ac.uk/fac/sci/dcs/research/tia/data/pannuke/fold_{fold_ix}.zip"
        r = requests.head(url)
        # HTTP status code 200 means "OK"
        assert r.status_code == 200


def test_zero_division():
    with pytest.raises(AssertionError):
        pannuke = PanNukeDataModule(data_dir = "wrong/path/to/pannuke", download = False)