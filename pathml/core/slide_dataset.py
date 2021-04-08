from torch.utils.data import ConcatDataset, DataLoader


class SlideDataSet:
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

    @property
    def tile_dataset(self):
        """
        Returns:
            torch.utils.data.Dataset: A PyTorch Dataset object of preprocessed tiles
        """
        return self._tile_dataset
