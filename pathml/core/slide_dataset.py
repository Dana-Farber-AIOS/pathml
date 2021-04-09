from torch.utils.data import ConcatDataset
from typing import List
from pathlib import Path


class SlideDataset:
    """
    Container for a dataset of WSIs

    Args:
        slides: list of SlideData objects
    """
    def __init__(self, slides):
        self.slides = slides
        self._tile_dataset = None

    def __getitem__(self, ix):
        return self.slides[ix]

    def __len__(self):
        return len(self.slides)

    def run(self, pipeline, client=None, tile_size=3000, tile_stride=None, level=0, tile_pad=False):
        """
        Runs a preprocessing pipeline on all slides in the dataset

        Args:
            Args:
            pipeline (pathml.preprocessing.pipeline.Pipeline): Preprocessing pipeline.
            tile_size (int, optional): Size of each tile. Defaults to 3000px
            tile_stride (int, optional): Stride between tiles. If ``None``, uses ``tile_stride = tile_size``
                for non-overlapping tiles. Defaults to ``None``.
            level (int, optional): Level to extract tiles from. Defaults to ``None``.
            tile_pad (bool): How to handle chunks on the edges. If ``True``, these edge chunks will be zero-padded
                symmetrically and yielded with the other chunks. If ``False``, incomplete edge chunks will be ignored.
                Defaults to ``False``.
        """
        # run preprocessing
        for slide in self.slides:
            slide.run(pipeline, client=client, tile_size=tile_size,
                      tile_stride=tile_stride, level=level, tile_pad=tile_pad)

        assert not any([s.tile_dataset is None for s in self.slides])
        # create a tile dataset for the whole dataset
        self._tile_dataset = ConcatDataset([s.tile_dataset for s in self.slides])

    def reshape(self, shape, centercrop=False):
        for slide in self.slides:
            slide.tiles.reshape(shape = shape, centercrop = centercrop)

    def write(self, dir, filenames=None):
        """
        Write all SlideData objects to the specified directory.
        Calls .write() method for each slide in the dataset. Optionally pass a list of filenames to use,
        otherwise filenames will be created from ``.name`` attributes of each slide.

        Args:
            dir (Union[str, bytes, os.PathLike]): Path to directory where slides are to be saved
            filenames (List[str], optional): list of filenames to be used.
        """
        d = Path(dir)
        if filenames:
            if len(filenames) != self.__len__():
                raise ValueError(f"input list of filenames has {len(filenames)} elements "
                                 f"but must be same length as number of slides in dataset ({self.__len__()})")

        for i, slide in enumerate(self.slides):
            if filenames:
                slide_path = d / (filenames[i] + ".h5path")
            elif slide.name:
                slide_path = d / (slide.name + ".h5path")
            else:
                raise ValueError("slide does not have a .name attribute. Must supply a 'filenames' argument.")
            slide.write(slide_path)

    @property
    def tile_dataset(self):
        """
        Returns:
            torch.utils.data.Dataset: A PyTorch Dataset object of preprocessed tiles
        """
        return self._tile_dataset
