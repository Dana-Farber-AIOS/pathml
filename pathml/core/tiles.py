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
    # TODO: 
    # 1. do we want to be able to tiles[tile].masks, tiles[tile].masks.add(), tiles[tile].masks.remove()
    #    at the moment we getitem the whole tile object, modify it, add it back
    # 2. connected to ^ we keep a copy of masks in the reference to the tile (redundant)
    # both of these problems are connected to the question whether h5 should be one file (we could hold 
    # reference to tile.masks giving us (1) but this gives us 2+ objects)
    # 3. label type
    """
    Object holding tiles.

    Args:
        tiles (Union[dict[tuple[int], `~pathml.core.tiles.Tile`], list]): tile objects  
    """
    def __init__(self,
            tiles =  None
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
                    if not isinstance(tile, Tile):
                        raise ValueError(f"Tiles expects a list of type Tile but was given {type(tile)}")
                    tiledictionary[(tile.i, tile.j)] = tiles[tile]
                self._tiles = OrderedDict(tiledictionary)
        else:
            self._tiles = OrderedDict()
        self.h5manager = _tiles_h5_manager() 
        for key in self._tiles:
            self.h5manager.add(key, self._tiles[key])
        del self._tiles

    def __repr__(self):
        rep = f"Tiles(keys={self.h5manager.h5.keys()})"
        return rep

    def __len__(self):
        return len(self.h5manager.h5['tiles'].keys())

    def __getitem__(self, item):
        name, tile, maskdict, labels = self.h5manager.get(item) 
        if isinstance(item, tuple):
            return Tile(tile, masks=Masks(maskdict), labels=labels, i=item[0], j=item[1]) 
        return Tile(tile, masks=Masks(maskdict), labels=labels)

    def add(self, coordinates, tile):
        """
        Add tile indexed by coordinates to self.h5manager.

        Args:
            coordinates(tuple[int]): location of tile on slide
            tile(Tile): tile object
        """
        if not isinstance(tile, Tile):
            raise ValueError(f"can not add {type(tile)}, tile must be of type pathml.core.tiles.Tile")
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
        Path(out_dir).mkdir(parents=True, exist_ok=True) 
        newfile = os.path.abspath(str(savepath.with_suffix('.h5')))
        newh5 = h5py.File(newfile, 'a')

        #shutil.move(self.h5manager.h5path, newh5)
        for dataset in self.h5manager.h5.keys():
            self.h5manager.h5.copy(self.h5manager.h5[dataset], newh5)

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
        assert isinstance(masks, (type(None), Masks, dict)), f"masks is of type {type(masks)} but must be of type pathml.core.masks.Masks or dict"
        if isinstance(masks, Masks):
            self.masks = masks
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

if __name__ == '__main__':
    import random
    import string
    maskdict = {}
    letters = string.ascii_letters + string.digits
    for i in range(50):
        randomkey = 'test' + ''.join(random.choice(letters) for j in range(i))
        maskdict[randomkey] = np.random.randint(2, size=(224,224,3))
    masks = Masks(maskdict)
    testtile = Tile(np.random.random_sample((224,224,3)), i=3, j=4, masks=masks)
    tiles = Tiles()
    tiles.add((2, 4), testtile)
    print(tiles.h5manager.h5.keys())
    print(tiles.h5manager.h5['(2, 4)'].keys())
    print(tiles.h5manager.h5['(2, 4)']['masks'].keys())
    print(tiles[(2, 4)].labels)
    print(tiles[0])
    tiles.remove((2,4))
    print(tiles)
