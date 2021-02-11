import numpy as np
import os
from pathlib import Path
from collections import OrderedDict
import h5py

from pathml.core.h5managers import _masks_h5_manager


class Masks:
    """
    Object holding masks.

    Args:
        masks(dict): Mask objects representing ex. labels, segmentations.
    """

    def __init__(self,
                 masks=None
                 ):
        if masks:
            if not isinstance(masks, dict):
                raise ValueError(f"masks must be passed as dicts of the form key1:mask1,key2:mask2,...")
            for val in masks.values():
                if not isinstance(val, np.ndarray):
                    raise ValueError(f"can not add {type(val)}, mask must be of type np.ndarray")
            for key in masks.keys():
                if not isinstance(key, str):
                    raise ValueError(f"can not add {type(key)}, key must be of type str")
            self._masks = OrderedDict(masks)
        else:
            self._masks = OrderedDict()
        self.h5manager = _masks_h5_manager()
        for mask in self._masks:
            self.h5manager.add(mask, self._masks[mask])
        del self._masks

    def __repr__(self):
        rep = f"Masks(keys={self.h5manager.h5['masks'].keys()})"
        return rep

    def __len__(self):
        return len(self.h5manager.h5['masks'].keys())

    def __getitem__(self, item):
        return self.h5manager.get(item)

    def __setitem__(self, key, mask):
        self.h5manager.update(key, mask)

    def add(self, key, mask):
        """
        Add mask indexed by key to self.h5manager.

        Args:
            key (str): key
            mask (np.ndarray): array of mask. Must contain elements of type int8
        """
        self.h5manager.add(key, mask)

    def slice(self, slices):
        """
        Slice all masks in self.h5manager extending of numpy array slicing.

        Args:
            slices: list where each element is an object of type slice indicating
                    how the dimension should be sliced
        """
        if not (isinstance(slices, list) and all([isinstance(a, slice) for a in slices])):
            raise KeyError(f"slices must of of type list[slice] but is {type(slices)} with elements {type(slices[0])}")
        sliced = Masks()
        for key, val in self.h5manager.slice(slices):
            sliced.add(key, val[slices])
        return sliced

    def remove(self, key):
        """
        Remove mask from self.h5manager by key.
        """
        self.h5manager.remove(key)

    def resize(self, shape):
        raise NotImplementedError

    def write(self, out_dir, filename):
        """
        Save masks as .h5 

        Args:
            out_dir(str): directory to write
            filename(str) file name
        """
        savepath = Path(out_dir) / Path(filename)
        Path(out_dir).mkdir(parents = True, exist_ok = True)
        # abspath resolves documented h5py bug
        newfile = os.path.abspath(str(savepath.with_suffix('.h5')))
        newh5 = h5py.File(newfile, 'w')

        # shutil.move(self.h5manager.h5path, newh5)
        for dataset in self.h5manager.h5.keys():
            self.h5manager.h5.copy(self.h5manager.h5[dataset], newh5)

    def read(self, path):
        """
        Read masks from .h5
        """
        raise NotImplementedError
