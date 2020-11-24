from torch.utils import data as data


class BaseSlideDataset(data.Dataset):
    """
    A base class for a collection of whole-slide-images.
    Specific datasets should inherit from this class.
    """
    def __len__(self):
        # This should return the size of the dataset
        raise NotImplementedError

    def __getitem__(self, ix):
        # this should return a (slide, label) pair for each index
        raise NotImplementedError


class BaseTileDataset(data.Dataset):
    """
    A base class for a collection of tiles.
    Specific datasets should inherit from this class.
    """
    def __len__(self):
        # This should return the size of the dataset
        raise NotImplementedError

    def __getitem__(self, ix):
        # this should return a (tile, label) pair for each index
        # can also do things like (tile, tile-label, slide-label)
        raise NotImplementedError


class BaseDataModule:
    """
    A base class for a DataModule.
    DataModules perform all the steps needed for a dataset, from downloading the data to creating dataloaders.
    Specific DataModules should inherit from this class.
    Inspired by pytorch-lightning LightningDataModule
    """
    def train_dataloader(self):
        raise NotImplementedError

    def valid_dataloader(self):
        raise NotImplementedError

    def test_dataloader(self):
        raise NotImplementedError
