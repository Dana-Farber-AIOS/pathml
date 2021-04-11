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
