import concurrent.futures
import os

from pathml.preprocessing.base import BaseSlideLoader, BaseSlidePreprocessor, BaseTileExtractor, BaseTilePreprocessor, \
    BasePipeline, BaseSlide
import pickle

from pathml.core.slide import BaseSlide
from pathml.datasets.base import BaseDataset


def save(self, filename):
        """
        save pipeline by writing them to disk
        :param filename: save path on disk
        :type path: str
        :return: string indicated file saved to above path
        """
        pickle.dump(self, open(filename, "wb"))
        return filename


class BasePipeline:
    """
    Base class for Pipeline objects
    """

    def run_single(self, slide, **kwargs):
        """
        Define pipeline here for a single BaseSlide object
        """
        raise NotImplementedError

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