import pytest

from pathml.datasets.pannuke import PanNukeDataModule


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
