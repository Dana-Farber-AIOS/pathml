"""
Copyright 2021, Dana-Farber Cancer Institute and Weill Cornell Medicine
License: GNU GPL 2.0
"""

import reprlib
from pathlib import Path
from loguru import logger

import dask.distributed
from torch.utils.data import ConcatDataset


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

    def __repr__(self):
        out = []
        out.append(f"SlideDataset object with {len(self)} slides")
        out.append(f"names: {reprlib.repr([s.name for s in self.slides])}")
        out.append(f"shapes: {reprlib.repr([s.shape for s in self.slides])}")

        out = ",\n\t".join(out)
        out += ")"
        return out

    def run(self, pipeline, client=None, distributed=True, **kwargs):
        """
        Runs a preprocessing pipeline on all slides in the dataset

        Args:
            pipeline (pathml.preprocessing.pipeline.Pipeline): Preprocessing pipeline.
            client: dask.distributed client
            distributed (bool): Whether to distribute model using client. Defaults to True.
            kwargs (dict): keyword arguments passed to :meth:`~pathml.core.slide_data.SlideData.run` for each slide
        """
        shutdown_after = False
        # run preprocessing
        if client is None and distributed:
            client = dask.distributed.Client()
            shutdown_after = True
        for slide in self.slides:
            slide.run(
                pipeline=pipeline, client=client, distributed=distributed, **kwargs
            )
        if shutdown_after:
            client.shutdown()

    def reshape(self, shape, centercrop=False):
        for slide in self.slides:
            slide.tiles.reshape(shape=shape, centercrop=centercrop)

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
                raise ValueError(
                    logger.exception(
                        f"input list of filenames has {len(filenames)} elements but must be same length as number of slides in dataset ({self.__len__()})"
                    )
                )

        for i, slide in enumerate(self.slides):
            if filenames:
                slide_path = d / (filenames[i] + ".h5path")
            elif slide.name:
                slide_path = d / (slide.name + ".h5path")
            else:
                raise ValueError(
                    logger.exception(
                        f"slide does not have a .name attribute. Must supply a 'filenames' argument."
                    )
                )
            slide.write(slide_path)
