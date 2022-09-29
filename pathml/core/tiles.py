"""
Copyright 2021, Dana-Farber Cancer Institute and Weill Cornell Medicine
License: GNU GPL 2.0
"""

import os
import reprlib
from collections import OrderedDict
from pathlib import Path

import h5py
from loguru import logger

import pathml.core.h5managers
import pathml.core.masks
import pathml.core.tile


class Tiles:
    """
    Object wrapping a dict of tiles.

    Args:
        tiles (Union[dict[tuple[int], `~pathml.core.tiles.Tile`], list[`~pathml.core.tiles.Tile`]]): tile objects
    """

    def __init__(self, h5manager, tiles=None):
        assert isinstance(h5manager, pathml.core.h5managers.h5pathManager)
        self.h5manager = h5manager
        # if tiles are supplied, add them to the h5manager
        if tiles:
            assert isinstance(tiles, list) and all(
                [isinstance(tile, pathml.core.Tile) for tile in tiles]
            ), f"tiles are of type {reprlib.repr([type(t) for t in tiles])} but must all be pathml.core.Tile"

            tiledictionary = {}
            for tile in tiles:
                if not isinstance(tile, pathml.core.Tile):
                    raise ValueError(
                        f"Tiles expects a list of type Tile but was given {type(tile)}"
                    )
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
        return self.h5manager.h5["tiles"].attrs["tile_shape"]

    @property
    def keys(self):
        return list(self.h5manager.h5["tiles"].keys())

    def __repr__(self):
        rep = f"{len(self.h5manager.h5['tiles'])} tiles: {reprlib.repr(list(self.h5manager.h5['tiles'].keys()))}"
        return rep

    def __len__(self):
        return len(self.h5manager.h5["tiles"].keys())

    def __getitem__(self, item):
        return self.h5manager.get_tile(item)

    def add(self, tile):
        """
        Add tile indexed by tile.coords to tiles.

        Args:
            tile(Tile): tile object
        """
        if not isinstance(tile, pathml.core.tile.Tile):
            raise ValueError(
                f"can not add {type(tile)}, tile must be of type pathml.core.tiles.Tile"
            )
        self.h5manager.add_tile(tile)

    def update(self, tile):
        """
        Update a tile.

        Args:
            tile(pathml.core.tile.Tiles): key of tile to be updated
        """
        self.h5manager.add_tile(tile)

    def remove(self, key):
        """
        Remove tile from tiles.

        Args:
            key(str): key (coords) indicating tile to be removed
        """
        self.h5manager.remove_tile(key)
