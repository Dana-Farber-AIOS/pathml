"""
Copyright 2021, Dana-Farber Cancer Institute and Weill Cornell Medicine
License: GNU GPL 2.0
"""

import os
from pathlib import Path
from collections import OrderedDict
import h5py
import reprlib

import pathml.core.h5managers
import pathml.core.tile
import pathml.core.masks


class Tiles:
    """
    Object wrapping a dict of tiles.

    Args:
        tiles (Union[dict[tuple[int], `~pathml.core.tiles.Tile`], list[`~pathml.core.tiles.Tile`]]): tile objects
    """
    def __init__(self, h5manager, tiles=None, slide_type=None):
        self.h5manager = h5manager 

        # if tiles are supplied, add them to the h5manager
        if tiles:
            if not (isinstance(tiles, dict) or (isinstance(tiles, list) and all([isinstance(t, pathml.core.tile.Tile) for t in tiles]))):
                raise ValueError(f"tiles must be passed as dicts of the form coordinates1:Tile1,... "
                                 f"or lists of Tile objects containing coords")
            # create _tiles from dict of tile
            if isinstance(tiles, dict):
                for val in tiles.values():
                    if not isinstance(val, pathml.core.Tile):
                        raise ValueError(f"tiles dict supplied, but got a value of type {type(val)}. Must be Tile")
                for key in tiles.keys():
                    if not (isinstance(key, tuple) and all(isinstance(v, int) for v in key)):
                        raise ValueError(f"dict keys must be of type tuple[int]")
                self._tiles = OrderedDict(tiles)

            # create _tiles from list of tile
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

            # add tiles in _tiles to h5manager
            for key, val in self._tiles.items():
                self.h5manager.add_tile(val)
            del self._tiles

    @property
    def tile_shape(self):
        return eval(self.h5manager.h5["tiles"].attrs["tile_shape"])

    def __repr__(self):
        rep = f"Tiles object with {len(self.h5manager.h5['tiles'])} tiles: {reprlib.repr(list(self.h5manager.h5['tiles'].keys()))}"
        return rep

    def __len__(self):
        return len(self.h5manager.h5["tiles"].keys())

    def __getitem__(self, item):
        tile = self.h5manager.get_tile(item) 
        return tile 

    def add(self, tile):
        """
        Add tile indexed by tile.coords to tiles.

        Args:
            tile(Tile): tile object
        """
        if not isinstance(tile, pathml.core.tile.Tile):
            raise ValueError(f"can not add {type(tile)}, tile must be of type pathml.core.tiles.Tile")
        self.h5manager.add_tile(tile)
        del tile

    def update(self, key, val, target='all'):
        """
        Update a tile.

        Args:
            key(str): key of tile to be updated
            val(str): element that will replace target at key
            target(str): element of {all, image, labels} indicating field to be updated 
        """
        self.h5manager.update_tile(key, val, target)

    def slice(self, slicer):
        """
        Generator slicing all tiles, extending numpy array slicing.

        Args:
            slicer: List where each element is an object of type slice https://docs.python.org/3/c-api/slice.html
                    indicating how the corresponding dimension should be sliced. The list length should correspond to the
                    dimension of the tile. For 2D H&E images, pass a length 2 list of slice objects.

        Yields:
            key(str): tile coordinates
            val(pathml.core.tile.Tile): tile
        """
        if not (isinstance(slicer,list) and (isinstance(a,slice) for a in slicer)):
            raise KeyError(f"slices must of of type list[slice]")
        sliced = [tile for tile in self.h5manager.slice_tiles(slicer)] 
        return sliced

    def remove(self, key):
        """
        Remove tile from tiles.

        Args:
            key(str): key (coords) indicating tile to be removed
        """
        self.h5manager.remove_tile(key)

    def reshape(self, shape, centercrop = False):
        """
        Resample tiles to shape. 
        If shape does not evenly divide current tile shape, this method deletes tile labels and names.
        This method not mutate h5['tiles']['array'].

        Args:
            shape(tuple): new shape of tile.
            centercrop(bool): if shape does not evenly divide slide shape, take center crop
        """
        assert isinstance(shape, tuple) and all(isinstance(n, int) for n in shape)
        assert isinstance(centercrop, bool)
        self.h5manager.reshape_tiles(shape, centercrop)
