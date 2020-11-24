import numpy as np
from collections import OrderedDict

class Masks():
    '''
    Class wrapping OrderedDict of masks.
    Masks are type np.ndarray with elements type int8.
    '''
    def __init__(self):
        self._masks = OrderedDict()

    def __repr__(self):
        rep = f"Masks(keys={self._masks.keys()})"
        return rep

    def __len__(self):
        return len(self._masks)

    def __getitem__(self, idx):
        if idx > len(self._masks)-1:
            raise KeyError(f"index out of range [0,{len(self._masks)-1}]") 
        return list(self._masks.values())[idx]

    # @masks.setter instead?
    def add(self, key, mask):
        if not isinstance(mask, np.ndarray):
            raise Exception("mask must be of type np.ndarray")
        if not isinstance(key, str):
            raise Exception("key must be of type str")
        if key in self._masks:
            print(f"overwriting mask {key}")
        self._masks[key] = mask

    def remove(self, key):
        if key not in self._masks:
            raise KeyError('key is not in dict Masks')
        del self._masks[key]
    
    @property
    def masks(self):
        return self._masks
