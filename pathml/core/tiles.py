import numpy as np
import os
import cv2
import shutil
from typing import Union
from pathlib import Path
from collections import OrderedDict
import h5py

from pathml.core.masks import Masks
from pathml.core.h5managers import _tiles_h5_manager

class Tiles:
    """
    Object holding tiles.

    Args:
        tiles (Union[dict[tuple[int], `~pathml.core.tiles.Tile`], list]): tile objects  
    """
    def __init__(self,
            tiles = None
        ):
        if tiles:
            if not isinstance(tiles, (dict, list[Tile])):
                raise ValueError(f"tiles must be passed as dicts of the form coordinate1:Tile1,... or lists of Tile objects containing i,j")
            if isinstance(tiles, dict):
                for val in tiles.values():
                    if not isinstance(val, Tile):
                        raise ValueError(f"dict vals must be Tile")
                for key in tiles.values():
                    if not isinstance(key, tuple[int]):
                        raise ValueError(f"dict keys must be tuple[int]")
                self._tiles = OrderedDict(tiles)
            else:
                tiledictionary = {}
                for tile in tiles:
                    tiledictionary[(tile.i, tile.j)] = tiles[tile]
                self._tiles = OrderedDict(tiledictionary)
        else:
            self._tiles = OrderedDict()
        self.h5manager = _tiles_h5_manager() 
        for key in self._tiles:
            self.h5manager.add(key, self._tiles[key])
            del self._tiles[key]

    def __repr__(self):
        rep = f"Tiles(keys={self.h5manager.h5['tiles'].keys()})"
        return rep

    def __len__(self):
        return len(self.h5manager.h5['tiles'].keys())

    def __getitem__(self, item):
        return self.h5manager.get(item) 

    def add(self, coordinates, tile):
        """
        Add tile indexed by coordinates to self.h5manager.

        Args:
            coordinates(tuple[int]): location of tile on slide
            tile(Tile): tile object
        """
        if not isinstance(tile, Tile):
            raise ValueError(f"can not add {type(tile)}, tile must be of type pathml.core.tiles.Tile")
        if not isinstance(coordinates, tuple):
            raise ValueError(f"can not add type {type(key)}, key must be of type tuple[int]")
        self.h5manager.add(coordinates, tile)

    def slice(self, coordinates):
        """
        Slice all tiles in self.h5manager extending numpy array slicing

        Args:
            coordinates(tuple[int]): coordinates denoting slice i.e. 'selection' https://numpy.org/doc/stable/reference/arrays.indexing.html
        """
        sliced = Tiles()
        for key, val in self.h5manager.slice(coordinates):
            sliced.add(key, val)
        return sliced

    def remove(self, key):
        """
        Remove tile from self.h5manager by key.
        """
        self.h5manager.remove(key)

    def resize(self, shape):
        raise NotImplementedError

    def write(self, out_dir, filename):
        """
        Save tiles as .h5 

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

    def read(self, path):
        """
        Read tiles from .h5
        """
        raise NotImplementedError



class Tile:
    """
    Object representing a tile extracted from an image. Holds the image for the tile, as well as the (i,j)
    coordinates of the top-left corner of the tile in the original image. The (i,j) coordinate system is based
    on labelling the top-leftmost pixel as (0, 0)

    Args:
        array (np.ndarray): tile image 
        masks (dict, Masks): tile masks 
        i (int): vertical coordinate of top-left corner of tile, in original image
        j (int): horizontal coordinate of top-left corner of tile, in original image
    """
    def __init__(self, array, labels=None, masks=None, i=None, j=None):
        assert isinstance(array, np.ndarray), "Array must be a np.ndarray"
        self.array = array
        self.shape = array.shape
        self.i = i  # i coordinate of top left corner pixel
        self.j = j  # j coordinate of top left corner pixel
        assert isinstance(masks, (type(None), Masks, dict)), f"masks is of type {type(masks)} but must be of type pathml.core.Masks or dict"
        if isinstance(masks, Masks):
            self.masks = Masks
        # populate Masks object by dict
        if isinstance(masks, dict): 
            for val in masks.values():
                if val.shape != self.array.shape[:2]:
                    raise ValueError(f"mask is of shape {val.shape} but must match tile shape {self.array.shape}")
            self.masks = Masks(masks)
        elif masks == None:
            self.masks = masks 
        assert isinstance(labels, (type(None), dict))
        self.labels = labels

    def __repr__(self):  # pragma: no cover
        return f"Tile(array shape {self.array.shape}, " \
               f"i={self.i if self.i is not None else 'None'}, " \
               f"j={self.j if self.j is not None else 'None'})"


if __name__ == "__main__":
    # create tile
    tiles = Tiles()
    testtile = Tile(np.ones((224,224,3)), i=1, j=3)
    # add to tiles .h5
    tiles.add((1,3), testtile)
    print(tiles)
    # write temp .h5 to persistent .h5
    tiles.write('out','test.h5')
    # remove tiles
    tiles.remove('(1, 3)')
    print(tiles)
    test = h5py.File('out/test.h5')
    print(test['tiles']['(1, 3)'][:])
