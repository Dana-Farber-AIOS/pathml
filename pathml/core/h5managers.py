import h5py
import tempfile

import numpy as np

class _tiles_h5_manager:
    """
    Interface between tiles object and data management on disk by h5py. 
    """
    def __init__(self):
        path = tempfile.TemporaryFile()
        f = h5py.File(path, 'w')
        self.h5 = f
        self.h5path = path
        self.shape = None

    def add(self, coordinates, tile):
        """
        Add tile as dataset indexed by coordinate to self.h5.

        Args:
            coordinates(tuple[int]): location of tile on slide
            tile(`~pathml.core.tile.Tile`): Tile object 
        """
        if not isinstance(coordinates, tuple):
            raise ValueError(f"can not add type {type(key)}, key must be of type tuple[int]")
        if str(coordinates) in self.h5.keys():
            print(f"overwriting tile at {coordinates}")
        if self.shape == None:
            self.shape = tile.array.shape
        if tile.array.shape != self.shape:
            raise ValueError(f"Tiles contains tiles of shape {self.shape}, provided tile is of shape {tile.array.shape}. We enforce that all Tile in Tiles must have matching shapes.")
        tilegroup = self.h5.create_group(str(coordinates))
        masksgroup = tilegroup.create_group('masks')
        labelsgroup = tilegroup.create_group('labels')
        addtile = tilegroup.create_dataset(
            'tile',
            data = tile.array
        )
        if tile.masks:
            for mask in tile.masks.h5manager.h5['masks']: 
                addmask = masksgroup.create_dataset(
                        str(mask),
                        data = tile.masks[mask]
                )
        if tile.labels:
            addlabels = labelsgroup.create_dataset(
                    'labels',
                    data = np.array(tile.labels, dtype='S')
            )

    def slice(self, coordinates, slicedict):
        """
        Generator to slice all tiles in self.h5 extending numpy array slicing

        Args:
            coordinates(tuple[int]): coordinates denoting slice i.e. 'selection' https://numpy.org/doc/stable/reference/arrays.indexing.html

        Yields:
            key(str): tile coordinates
            val(`~pathml.core.tile.Tile`): tile
        """
        raise NotImplementedError
        # how to pass numpy indexing style from tiles
        for key, val in self.h5.items():
            val = val[coordinates]
            yield key, val

    def get(self, item):
        if isinstance(item, tuple):
            if str(item) not in self.h5.keys():
                raise KeyError('key {item} does not exist')
            tile = self.h5[str(item)]['tile'][:]
            maskdict = {key:self.h5[str(item)]['masks'][key][:] for key in self.h5[str(item)]['masks'].keys()}
            # TODO: decide on type for labels so they can be read back to tile
            labels = None
            return item, tile, maskdict, labels
        if not isinstance(item, int):
            raise KeyError(f"must getitem by coordinate(type tuple[int]) or index(type int)")
        if item > len(self.h5)-1:
            raise KeyError(f"index out of range, valid indices are ints in [0,{len(self.h5['tiles'].keys())-1}]")
        tile = self.h5[list(self.h5.keys())[item]]['tile'][:]
        maskdict = {key:self.h5[list(self.h5.keys())[item]]['masks'][key][:] for key in self.h5[list(self.h5.keys())[item]]['masks'].keys()} 
        # TODO: decide on type for labels so they can be read back to tile
        labels = None
        return list(self.h5.keys())[item], tile, maskdict, labels

    def remove(self, key):
        """
        Remove tile from self.h5 by key.
        """
        if not isinstance(key, (str,tuple)):
            raise KeyError(f'key must represent tuple, check valid keys in repr')
        if str(key) not in self.h5.keys():
            raise KeyError(f'key {key} is not in Tiles')
        del self.h5[str(key)]

class _masks_h5_manager:
    """
    Interface between masks object and data management on disk by h5py. 
    """
    def __init__(self):
        path = tempfile.TemporaryFile()
        f = h5py.File(path, 'w')
        f.create_group("masks")
        self.h5 = f
        self.h5path = path
        self.shape = None

    def add(self, key, mask):
        """
        Add mask as dataset indexed by key to self.h5.

        Args:
            key(str): key labeling mask
            mask(np.ndarray): mask  
        """
        if not isinstance(mask, np.ndarray):
            raise ValueError(f"can not add {type(mask)}, mask must be of type np.ndarray")
        if not isinstance(key, str):
            raise ValueError(f"invalid type {type(key)}, key must be of type str")
        if key in self.h5['masks'].keys():
            print(f"overwriting key at {key}")
        if self.shape == None:
            self.shape = mask.shape
        if mask.shape != self.shape:
            raise ValueError(f"Masks contains masks of shape {self.shape}, provided mask is of shape {mask.shape}. We enforce that all Mask in Masks must have matching shapes.")
        newkey = self.h5['masks'].create_dataset(
            bytes(str(key), encoding='utf-8'),
            data = mask
        )

    def slice(self, coordinates):
        """
        Generator to slice all masks in self.h5 extending numpy array slicing

        Args:
            coordinates(tuple[int]): coordinates denoting slice i.e. 'selection' https://numpy.org/doc/stable/reference/arrays.indexing.html

        Yields:
            key(str): mask key
            val(np.ndarray): mask
        """
        for key, val in self.h5.items():
            val = val[coordinates]
            yield key, val

    def get(self, item):
        if isinstance(item, str):
            if item not in self.h5['masks'].keys():
                raise KeyError(f'key {item} does not exist')
            return self.h5['masks'][item][:]
        if not isinstance(item, int):
            raise KeyError(f"must getitem by name (type str) or index(type int)")
        if item > len(self.h5['masks'])-1:
            raise KeyError(f"index out of range, valid indices are ints in [0,{len(self.h5['masks'].keys())-1}]")
        return self.h5['masks'][list(self.h5['masks'].keys())[item]][:]

    def remove(self, key):
        """
        Remove mask from self.h5 by key.
        """
        if not isinstance(key, str):
            raise KeyError(f"masks keys must be of type(str) but key was passed of type {type(key)}")
        if key not in self.h5['masks'].keys():
            raise KeyError('key is not in Masks')
        del self.h5['masks'][key]

def read_h5(path):
    raise NotImplementedError
    f = h5py.File(path, 'r+')
    if f['tiles']:
        pass
    if f['masks']:
        pass
