import os
from pathlib import Path
from collections import OrderedDict
import h5py

import pathml.core.h5managers
import pathml.core.tiles
import pathml.core.tile
import pathml.core.masks


class Tiles:
    """
    Object wrapping a dict of tiles.

    Args:
        tiles (Union[dict[tuple[int], `~pathml.core.tiles.Tile`], list[`~pathml.core.tiles.Tile`]]): tile objects  
    """
    def __init__(self, tiles=None, h5=None):
        # if h5, pass directly to h5manager
        if h5 is not None:
            self.h5manager = pathml.core.h5managers._tiles_h5_manager(h5)

        # no h5 supplied
        else:
            # initialize h5manager
            self.h5manager = pathml.core.h5managers._tiles_h5_manager()

            # if tiles are supplied, add them to the h5manager
            if tiles:
                if not (isinstance(tiles, dict) or (isinstance(tiles, list) and all([isinstance(t, pathml.core.tile.Tile) for t in tiles]))):
                    raise ValueError(f"tiles must be passed as dicts of the form coordinates1:Tile1,... "
                                     f"or lists of Tile objects containing coords")
                # create _tiles from dict of tile
                if isinstance(tiles, dict):
                    for val in tiles.values():
                        if not isinstance(val, pathml.core.tile.Tile):
                            raise ValueError(f"dict vals must be Tile")
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
                for key in self._tiles:
                    self.h5manager.add(self._tiles[key])
                del self._tiles

    def __repr__(self):
        rep = f"Tiles(keys={self.h5manager.tiles})"
        return rep

    def __len__(self):
        return len(self.h5manager.tiles.keys())

    def __getitem__(self, item):
        tile = self.h5manager.get(item) 
        return tile 

    def add(self, tile):
        """
        Add tile indexed by tile.coords to tiles.

        Args:
            tile(Tile): tile object
        """
        if not isinstance(tile, pathml.core.tile.Tile):
            raise ValueError(f"can not add {type(tile)}, tile must be of type pathml.core.tiles.Tile")
        self.h5manager.add(tile)
        del tile

    def update(self, key, val, target='all'):
        """
        Update a tile.

        Args:
            key(str): key of tile to be updated
            val(str): element that will replace target at key
            target(str): element of {all, image, labels} indicating field to be updated 
        """
        self.h5manager.update(key, val, target)

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
        sliced = pathml.core.tiles.Tiles()
        for tile in self.h5manager.slice(slicer):
            sliced.add(tile)
        return sliced

    def remove(self, key):
        """
        Remove tile from tiles.

        Args:
            key(str): key (coords) indicating tile to be removed
        """
        self.h5manager.remove(key)

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
        self.h5manager.reshape(shape, centercrop)
