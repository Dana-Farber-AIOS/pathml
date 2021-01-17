import numpy as np
from collections import OrderedDict

class Masks():
    '''
    Object holding masks.  

    :param masks: Mask objects representing ex. labels, segmentations. 
    :type masks: :class:`dict` with keys of type str and values of type np.ndarray with elements of type int8
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

    def __repr__(self):
        rep = f"Masks(keys={self._masks.keys()})"
        return rep

    def __len__(self):
        return len(self._masks)

    def __getitem__(self, item):
        if isinstance(item, str):
            return self._masks[item]
        if not isinstance(item, int):
            raise KeyError(f"must getitem by name(type str) or index(type int)")
        if item > len(self._masks)-1:
            raise KeyError(f"index out of range, valid indices are ints in [0,{len(self._masks)-1}]") 
        return list(self._masks.values())[item]

    def add(self, key, mask):
        """
        Add mask indexed by key to self._masks.

        :type key: str
        :type mask: np.ndarray with elements of type int8
        """
        if not isinstance(mask, np.ndarray):
            raise ValueError(f"can not add {type(mask)}, mask must be of type np.ndarray")
        if not isinstance(key, str):
            raise ValueError(f"invalid type {type(key)}, key must be of type str")
        if key in self._masks:
            print(f"overwriting mask {key}")
        if self._masks.keys():
            requiredshape = self._masks[list(self._masks.keys())[0]].shape
            if mask.shape != requiredshape:
                raise ValueError(f"masks must be of shape {requiredshape} but provided mask is of shape {mask.shape}") 
        self._masks[key] = mask

    def slice(self, coordinates):
        """
        Slice all masks in self._masks extending of numpy array slicing.

        :param coordinates: coordinates denoting slice i.e. 'selection' https://numpy.org/doc/stable/reference/arrays.indexing.html 
        """
        maskslice = Masks() 
        for key in self._masks.keys():
            val = self._masks[key]
            val = val[coordinates]
            maskslice.add(key, val)
        return maskslice

    def remove(self, key):
        """
        Remove mask from self._masks by key.
        """
        if key not in self._masks:
            raise KeyError('key is not in dict Masks')
        del self._masks[key]
    
    @property
    def masks(self):
        return self._masks
