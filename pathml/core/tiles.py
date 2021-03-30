import numpy as np
import os
import cv2
import shutil
from typing import Union
from pathlib import Path
from collections import OrderedDict
import h5py

from pathml.core.h5managers import _tiles_h5_manager
import pathml.core.tiles
import pathml.core.tile
import pathml.core.masks


class Tiles:
    """
    Object wrapping a dict of tiles.

    Args:
        tiles (Union[dict[tuple[int], `~pathml.core.tiles.Tile`], list[`~pathml.core.tiles.Tile`]]): tile objects  
    """
    def __init__(self, tiles = None, h5 = None):
        if h5 is None:
            if tiles:
                if not (isinstance(tiles, dict) or (isinstance(tiles, list) and all([isinstance(t, pathml.core.tile.Tile) for t in tiles]))):
                    raise ValueError(f"tiles must be passed as dicts of the form coordinates1:Tile1,... "
                                     f"or lists of Tile objects containing coords")
                # create Tiles from dict
                if isinstance(tiles, dict):
                    for val in tiles.values():
                        if not isinstance(val, pathml.core.tile.Tile):
                            raise ValueError(f"dict vals must be Tile")
                    for key in tiles.keys():
                        if not (isinstance(key, tuple) and all(isinstance(v, int) for v in key)):
                            raise ValueError(f"dict keys must be of type tuple[int]")
                    self._tiles = OrderedDict(tiles)
                # create Tiles from list
                else:
                    tiledictionary = {}
                    for tile in tiles:
                        if not isinstance(tile, pathml.core.tile.Tile):
                            raise ValueError(f"Tiles expects a list of type Tile but was given {type(tile)}")
                        if tile.coords is None:
                            raise ValueError(f"tiles must contain valid coords")
                        coords = tile.coords
                        tiledictionary[coords] = tile 
                    self._tiles = OrderedDict(tiledictionary)
            else:
                self._tiles = OrderedDict()
            # initialize h5 from tiles 
            self.h5manager = _tiles_h5_manager() 
            for key in self._tiles:
                self.h5manager.add(self._tiles[key])
            del self._tiles
        else:
            self.h5manager = _tiles_h5_manager(h5)

    def __repr__(self):
        rep = f"Tiles(keys={self.h5manager.h5.keys()})"
        return rep

    def __len__(self):
        return len(self.h5manager.tilesdict.keys())

    def __getitem__(self, item):
        tile = self.h5manager.get(item) 
        return tile 

    def add(self, tile):
        """
        Add tile indexed by tile.coords to self.h5manager.

        Args:
            tile(Tile): tile object
        """
        if not isinstance(tile, pathml.core.tile.Tile):
            raise ValueError(f"can not add {type(tile)}, tile must be of type pathml.core.tiles.Tile")
        self.h5manager.add(tile)
        del tile

    def update(self, key, val, target='all'):
        self.h5manager.update(key, val, target)

    def slice(self, slices):
        """
        Slice all tiles in self.h5manager extending numpy array slicing

        Args:
            slices: list where each element is an object of type slice indicating
                    how the dimension should be sliced
        """
        if not (isinstance(slices,list) and (isinstance(a,slice) for a in slices)):
            raise KeyError(f"slices must of of type list[slice]")
        sliced = pathml.core.tiles.Tiles()
        for tile in self.h5manager.slice(slices):
            sliced.add(tile)
        return sliced

    def remove(self, key):
        """
        Remove tile from self.h5manager by key.
        """
        self.h5manager.remove(key)

    def reshape(self, shape, centercrop = False):
        """
        Reshape tiles.
        """
        assert isinstance(shape, tuple) and all(isinstance(n, int) for n in shape) 
        assert isinstance(centercrop, bool)
        self.h5manager.reshape(shape, centercrop)

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
