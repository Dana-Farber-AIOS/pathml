import numpy as np
import os
import cv2
import shutil
from typing import Union
from pathlib import Path
from collections import OrderedDict
import h5py

from pathml.core.tile import Tile
from pathml.core.masks import Masks
from pathml.core.h5managers import _tiles_h5_manager
from pathml.core.tile import Tile


class Tiles:
    """
    Object wrapping a dict of tiles.

    Args:
        tiles (Union[dict[tuple[int], `~pathml.core.tiles.Tile`], list[`~pathml.core.tiles.Tile`]]): tile objects  
    """
    def __init__(self, tiles=None):
        if tiles:
            if not (isinstance(tiles, dict) or (isinstance(tiles, list) and all([isinstance(t, Tile) for t in tiles]))):
                raise ValueError(f"tiles must be passed as dicts of the form coordinate1:Tile1,... "
                                 f"or lists of Tile objects containing i,j")
            # create Tiles from dict
            if isinstance(tiles, dict):
                for val in tiles.values():
                    if not isinstance(val, Tile):
                        raise ValueError(f"dict vals must be Tile")
                for key in tiles.keys():
                    if not ((isinstance(key, tuple) and list(map(type, key)) == [int, int]) or isinstance(key, str)):
                        raise ValueError(f"dict keys must be of type str or tuple[int]")
                self._tiles = OrderedDict(tiles)
            # create Tiles from list
            else:
                tiledictionary = {}
                for tile in tiles:
                    if not isinstance(tile, Tile):
                        raise ValueError(f"Tiles expects a list of type Tile but was given {type(tile)}")
                    name = tile.name if tile.name is not None else str(tile.coords)
                    tiledictionary[name] = tile 
                self._tiles = OrderedDict(tiledictionary)
        else:
            self._tiles = OrderedDict()
        # move Tiles to .h5
        self.h5manager = _tiles_h5_manager() 
        for key in self._tiles:
            self.h5manager.add(key, self._tiles[key])
        del self._tiles

    def __repr__(self):
        rep = f"Tiles(keys={self.h5manager.h5.keys()})"
        return rep

    def __len__(self):
        return len(self.h5manager.h5.keys())

    def __getitem__(self, item):
        # TODO: Should move this logic into h5manager if possible. Fix recursive import problem
        name, tile, maskdict, labels, coords, slidetype = self.h5manager.get(item) 
        return Tile(tile, masks=Masks(maskdict), labels=labels, name=str(name), coords=coords, slidetype=slidetype)

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
        if not isinstance(slices,list) and (isinstance(slicer,slice) for slicer in slices):
            raise KeyError(f"slices must of of type list[slice] but is {type(slices)} with elements {type(slices[0])}")
        sliced = Tiles()
        for name, tile, maskdict, labels, coords, slidetype in self.h5manager.slice(slices):
            tile = Tile(image=tile, masks=Masks(maskdict), labels=labels, coords=coords, name=name, slidetype=slidetype)
            newimage = tile.image[slices]
            newmasks = {}
            for key, val in tile.masks.h5manager.h5['masks'].items():
                print(key, val)
                newmasks[str(key)] = val[:][slices]
            newtile = Tile(image=newimage, masks = newmasks, labels=labels, coords=coords, name=name, slidetype=slidetype) 

            sliced.add(name, newtile)
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
