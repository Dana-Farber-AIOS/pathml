import numpy as np
from collections import OrderedDict

from pathml.core.h5managers import _masks_h5_manager

class Masks:
    '''
    Object holding masks.  

    Args:
        masks(dict): Mask objects representing ex. labels, segmentations. 
    '''
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
        for mask in self._masks:
            self.h5manager.add(mask, self._masks[mask])
            del self._masks[mask]

    def __repr__(self):
        rep = f"Masks(keys={self._masks.keys()})"
        return rep

    def __len__(self):
        return len(self._masks)

    def __getitem__(self, item):
        if isinstance(item, str):
            return self.h5manager.h5[item]
        if not isinstance(item, int):
            raise KeyError(f"must getitem by name(type str) or index(type int)")
        if item > len(self._masks)-1:
            raise KeyError(f"index out of range, valid indices are ints in [0,{len(self._masks)-1}]") 
        return list(self.h5manager.h5.values())[item]

    def add(self, key, mask):
        """
        Add mask indexed by key to self.h5manager.

        :type key: str
        :type mask: np.ndarray with elements of type int8
        """
        if not isinstance(mask, np.ndarray):
            raise ValueError(f"can not add {type(mask)}, mask must be of type np.ndarray")
        if not isinstance(key, str):
            raise ValueError(f"invalid type {type(key)}, key must be of type str")
        self.h5manager.add(key, mask)

    def slice(self, coordinates):
        """
        Slice all masks in self.h5manager extending of numpy array slicing.
        Args:
            coordinates(tuple[int]): coordinates denoting slice i.e. 'selection' https://numpy.org/doc/stable/reference/arrays.indexing.html 
        """
        sliced = Masks()
        for key, val in self.h5manager.slice(coordinates):
            sliced.add(key, val)
        return sliced

    def remove(self, key):
        """
        Remove mask from self.h5manager by key.
        """
        self.h5manager.remove(key)

    def resize(self, shape):
        raise NotImplementedError

    def save(self, out_dir, filename):
        """
        Save masks as .h5 

        Args:
            out_dir(str): directory to write
            filename(str) file name
        """
        savepath = Path(out_dir) / Path(filename)
        try:
            os.mkdir(str(Path(out_dir)))
        except:
            pass
        newfile = f"{str(savepath.with_suffix('.h5'))}"
        newh5 = h5py.File(newfile, 'w')

        for dataset in self.h5manager.h5.keys():
            self.h5manager.h5.copy(self.h5manager.h5[dataset], newh5)
