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
        if str(coordinates) in self.h5.keys():
            print(f"overwriting tile at {coordinates}")
        if self.shape == None:
            self.shape = tile.array.shape
        newcoord = self.h5.create_dataset(
            str(coordinates),
            data = tile.array
        )
        if tile.array.shape != self.shape:
            raise ValueError(f"Tiles contains tiles of shape {self.shape}, provided tile is of shape {tile.array.shape}. We enforce that all Tile in Tiles must have matching shapes.")

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
        # how to pass coordinates?
        for key, val in self.h5.items():
            val = val[coordinates]
            yield key, val

    def get(self, item):
        if isinstance(item, tuple):
            if str(item) not in self.h5.keys():
                raise KeyError('key {index} does not exist')
            return self.h5[str(item)][:]
        if not isinstance(item, int):
            raise KeyError(f"must getitem by coordinate(type tuple[int]) or index(type int)")
        if item > len(self.h5.keys())-1:
            raise KeyError(f"index out of range, valid indeces are ints in [0,{len(self.h5.keys())-1}]")
        return self.h5[list(self.h5.keys())[item]][:]

    def remove(self, key):
        """
        Remove tile from self.h5 by key.
        """
        if key not in self.h5.keys():
            raise KeyError('key is not in Tiles')
        del self.h5[key]

class _masks_h5_manager:
    """
    Interface between masks object and data management on disk by h5py. 
    """
    def __init__(self):
        path = tempfile.TemporaryFile()
        f = h5py.File(path, 'w')
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
        if key in self.h5.keys():
            print(f"overwriting key at {key}")
        if self.h5.keys() is False:
            self.shape = mask.shape
        newkey = self.h5.create_dataset(
            str(key),
            data = mask,
            maxshape=(None, ) + mask.shape,
        )
        if mask.shape != self.shape:
            raise ValueError(f"Masks contains masks of shape {self.shape}, provided mask is of shape {mask.shape}. We enforce that all Mask in Masks must have matching shapes.")

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

    def remove(self, key):
        """
        Remove mask from self.h5 by key.
        """
        if key not in self.h5.keys():
            raise KeyError('key is not in Masks')
        del self.h5[key]
