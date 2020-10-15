import cv2
import torch.utils.data as data
from sklearn.model_selection import train_test_split
from warnings import warn
import numpy as np

from pathml.preprocessing.wsi import HESlide
from pathml.preprocessing.multiparametricslide import MultiparametricSlide


class SlideDataset(data.Dataset):
    """
    Dataset object for a collection of whole-slide-images
    """
    def __init__(self, slide_type, paths, labels):
        assert slide_type.lower() in ["he", "multiplex"], \
            f"ERROR: input slide type {slide_type} not supported. Must be one of ['he', 'multiplex']."
        self.slide_type = slide_type.lower()
        self.paths = paths
        self.labels = labels

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, ix):
        path = self.paths[ix]
        if self.slide_type == "he":
            slide = HESlide(path = path)
        elif self.slide_type == "multiplex":
            slide = MultiparametricSlide(path = path)
        else:
            raise Exception(f"ERROR: unrecognized slide_type (self.slide_type={self.slide_type})")
        label = self.labels[ix]
        return slide, label


class TileDataset(data.Dataset):
    """
    Dataset object for a collection of tiles.
    Supports both tile-level and slide-level labels.
    """
    def __init__(self, paths, tile_labels=None, slide_labels=None, transforms=None):
        self.paths = paths
        self.tile_labels = tile_labels
        self.slide_labels = slide_labels
        self.transforms = transforms

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, ix):
        path = self.paths[ix]
        im = cv2.imread(path)

        if self.transforms is not None:
            im = self.transforms(im)

        tile_label = None if self.tile_labels is None else self.tile_labels[ix]
        slide_label = None if self.slide_labels is None else self.slide_labels[ix]
        return im, tile_label, slide_label


class DataModule:
    """
    Data Module for handling datasets and creating dataloaders.
    File paths and labels are input in pandas DataFrames, and dataloaders are created for train, test, and validation
    sets at both slide-level and tile-level.

    Inspired by pytorch-lightning LightningDataModule

    Args:
        slide_anno (pd.DataFrame): Dataframe containing slide-level information.
            Each row is a slide.
            Must contain columns for "slide_id", "path", and "label".
            May optionally contain column for "split" specifying "train", "valid", or "test".
            If "split" column not included, then must specify split_fractions.
            If split_fractions is specified, then new splits will be generated and anything in "split"
            column will be overwritten.
        tile_anno (pd.DataFrame): DataFrame containing tile-level information.
            Each row is a tile.
            Must contain columns for "slide_id", "path", "tile_label", and "slide_label".
            May optionally contain column for "split" specifying "train", "valid", or "test".
            If "split" column not included, then must specify split_fractions.
            If split_fractions is specified, then new splits will be generated and anything in "split"
            column will be overwritten.
        tile_transforms: transforms to apply at tile-level
        batch_size (int): Batch size
        random_slide_split (bool): Whether to shuffle slides before splitting into train/test/validation sets.
            Ignored if split_fractions is None.
        shuffle_slides (bool): Whether to shuffle slides in dataloaders.
        shuffle_tiles (bool): Whether to shuffle tiles in dataloaders.
        random_seed (int): Random seed for shuffling. Ignored if split_fractions is None.
        split_fractions (tuple): Fractions for train/validation/test sets (at slide-level).
            Must be a tuple of three floats which sums to 1.
            If None, then both slide_anno and tile_anno must contain a "split" column.
    """
    def __init__(self, slide_type, slide_anno, tile_anno, tile_transforms, batch_size, random_slide_split,
                 shuffle_slides, shuffle_tiles, random_seed, split_fractions):

        # do checks of input conditions here
        if split_fractions is not None:
            assert np.isclose(1., np.sum(split_fractions)),\
                f"Error: input split_fractions {split_fractions} must sum to 1"
        if split_fractions is None:
            assert 'split' in slide_anno.columns and 'split' in tile_anno.columns, \
                "Error: split_fractions not specified and split columns not in slide_anno and tile_anno"
        assert "slide_id" in slide_anno.columns and "slide_id" in tile_anno.columns, \
            "Error: input slide_anno and tile_anno dataframes must both contain 'slide_id' column"
        assert "path" in slide_anno.columns and "path" in tile_anno.columns, \
            "Error: input slide_anno and tile_anno dataframes must both contain 'path' column"
        assert "label" in slide_anno.columns, "Error: input slide_anno dataframe must contain 'label' column"
        assert "tile_label" in slide_anno.columns, "Error: input slide_anno dataframe must contain 'tile_label' column"
        assert "slide_label" in slide_anno.columns, "Error: input slide_anno dataframe must contain 'slide_label' column"

        self.slide_type = slide_type
        self.slide_anno = slide_anno
        self.tile_anno = tile_anno
        self.tile_transforms = tile_transforms
        self.batch_size = batch_size
        self.random_slide_split = random_slide_split
        self.shuffle_slides = shuffle_slides
        self.shuffle_tiles = shuffle_tiles
        self.random_seed = random_seed
        self.split_fractions = split_fractions

        # create splits here if needed
        if split_fractions is not None:
            self.create_splits()

        # Set up SlideDatasets and TileDatasets from splits
        self.train_slide_dataset = SlideDataset(
            slide_type = self.slide_type,
            paths = self.slide_anno["path"].loc[self.slide_anno["split"] == "train"],
            labels = self.slide_anno["label"].loc[self.slide_anno["split"] == "train"]
        )
        self.valid_slide_dataset = SlideDataset(
            slide_type = self.slide_type,
            paths = self.slide_anno["path"].loc[self.slide_anno["split"] == "valid"],
            labels = self.slide_anno["label"].loc[self.slide_anno["split"] == "valid"]
        )
        self.test_slide_dataset = SlideDataset(
            slide_type = self.slide_type,
            paths = self.slide_anno["path"].loc[self.slide_anno["split"] == "test"],
            labels = self.slide_anno["label"].loc[self.slide_anno["split"] == "test"]
        )

        self.train_tile_dataset = TileDataset(
            paths = self.tile_anno["path"].loc[self.tile_anno["split"] == "train"],
            tile_labels = self.tile_anno["tile_label"].loc[self.tile_anno["split"] == "train"],
            slide_labels = self.tile_anno["slide_label"].loc[self.tile_anno["split"] == "train"],
            transforms = self.tile_transforms
        )
        self.valid_tile_dataset = TileDataset(
            paths = self.tile_anno["path"].loc[self.tile_anno["split"] == "valid"],
            tile_labels = self.tile_anno["tile_label"].loc[self.tile_anno["split"] == "valid"],
            slide_labels = self.tile_anno["slide_label"].loc[self.tile_anno["split"] == "valid"],
            transforms = self.tile_transforms
        )
        self.test_tile_dataset = TileDataset(
            paths = self.tile_anno["path"].loc[self.tile_anno["split"] == "test"],
            tile_labels = self.tile_anno["tile_label"].loc[self.tile_anno["split"] == "test"],
            slide_labels = self.tile_anno["slide_label"].loc[self.tile_anno["split"] == "test"],
            transforms = self.tile_transforms
        )

    def create_splits(self):
        """create train/validation/test splits"""
        # get size of each split
        frac_train, frac_valid, frac_test = self.split_fractions

        slide_ids = self.slide_anno["slide_id"].values

        # get slide_ids for train, test, validation sets
        trainvalid_ids, test_ids = train_test_split(slide_ids, test_size = frac_test,
                                                    random_state = self.random_seed, shuffle = self.random_slide_split)
        train_ids, valid_ids = train_test_split(
            trainvalid_ids,
            test_size = frac_valid / (frac_train + frac_valid),
            random_state = self.random_seed,
            shuffle = self.random_slide_split
        )

        # add split category to slide_anno dataframe
        if 'split' in self.slide_anno.colums:
            warn("Warning: 'split' column already exists in slide_anno dataframe. Overwriting with new splits.")
        self.slide_anno["split"] = ""
        for slide_id in slide_ids:
            if slide_id in test_ids:
                self.slide_anno.at[slide_id, "split"] = "test"
            elif slide_id in train_ids:
                self.slide_anno.at[slide_id, "split"] = "train"
            elif slide_id in valid_ids:
                self.slide_anno.at[slide_id, "split"] = "valid"
            else:
                Exception(f"Something went wrong in creating train/test/valid splits. Slide id {slide_id} not found.")

        # add split category to tile_anno dataframe
        if 'split' in self.tile_anno.colums:
            warn("Warning: 'split' column already exists in tile_anno dataframe. Overwriting with new splits.")
        self.tile_anno["split"] = self.tile_anno["slide_id"].apply(lambda x: self.slide_anno.at[slide_id, "split"])

    def train_slide_dataloader(self):
        return data.DataLoader(self.train_slide_dataset, shuffle = self.shuffle_slides)

    def valid_slide_dataloader(self):
        return data.DataLoader(self.valid_slide_dataset, shuffle = self.shuffle_slides)

    def test_slide_dataloader(self):
        return data.DataLoader(self.test_slide_dataset, shuffle = self.shuffle_slides)

    def train_tile_dataloader(self):
        return data.DataLoader(self.train_tile_dataset, shuffle = self.shuffle_tiles)

    def valid_tile_dataloader(self):
        return data.DataLoader(self.valid_tile_dataset, shuffle = self.shuffle_tiles)

    def test_tile_dataloader(self):
        return data.DataLoader(self.test_tile_dataset, shuffle = self.shuffle_tiles)
