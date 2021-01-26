import concurrent.futures
import os
import pickle

from pathml.core.slide import BaseSlide
from pathml.datasets.base import BaseDataset
from pathml.core.tile import Tile
from pathml.preprocessing.transforms import Transform


class Pipeline(Transform):
    """
    Compose a sequence of Transforms

    Args:
        transform_sequence (list): sequence of transforms to be consecutively applied.
            List of `pathml.core.Transform` objects
    """
    def __init__(self, transform_sequence):
        assert all([isinstance(t, Transform) for t in transform_sequence]), f"All elements in input list must be of" \
                                                                            f" type pathml.core.Transform"
        self.transforms = transform_sequence

    def __len__(self):
        return len(self.transforms)

    def __repr__(self):
        out = f"Pipeline([\n"
        for t in self.transforms:
            out += f"\t{repr(t)},\n"
        out += "])"
        return out

    def apply(self, tile):
        assert isinstance(tile, Tile), f"argument of type {type(tile)} must be a pathml.core.Tile object."
        for t in self.transforms:
            t.apply(tile)

    def save(self, filename):
        """
        save pipeline to disk

        Args:
            filename (str): save path on disk
        """
        pickle.dump(self, open(filename, "wb"))

    # TODO move this to be a method of SlideData
    def run(self, target, n_jobs=-1, **kwargs):
        """
        Execute pipeline on a single input or an entire dataset.
        If running on a dataset, this will use multiprocessing to distribute computation
        across available cores.

        Args:
            target (BaseSlide or BaseDataset): Input on which to execute pipeline.
            n_jobs (int): number of processes to use. If 1 or None, then no multiprocessing is used.
                If -1, then all available cores are used. Defaults to -1.
            kwargs (dict): Additional arguments passed to each individual call of self.run_single(target)
        """
        if isinstance(target, BaseSlide):
            # only need to run on a single input slide
            self.run_single(slide = target, **kwargs)
        elif isinstance(target, BaseDataset):
            # run on all elements in the dataset
            if n_jobs == 1 or n_jobs is None:
                # don't use multiprocessing in this case
                for slide in target:
                    self.run_single(slide = slide, **kwargs)

            else:
                if n_jobs == -1:
                    n_jobs = os.cpu_count()
                else:
                    assert isinstance(n_jobs, int), f"Input n_jobs {n_jobs} not valid. Must be None or an int"

                with concurrent.futures.ProcessPoolExecutor(max_workers = n_jobs) as executor:
                    for slide in target:
                        executor.submit(self.run_single, slide = slide, **kwargs)