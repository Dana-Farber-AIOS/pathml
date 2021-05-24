"""
Copyright 2021, Dana-Farber Cancer Institute and Weill Cornell Medicine
License: GNU GPL 2.0
"""

import h5py
import tempfile
import ast
from collections import OrderedDict
import numpy as np
import itertools

import pathml.core.masks
import pathml.core.tile
from pathml.core.utils import writetupleh5, readtupleh5, readtilesdicth5
import pathml.core.slide_data


class _h5_manager:
    """
    Abstract class for h5 data management.
    """
    def __init__(self, h5 = None):
        path = tempfile.TemporaryFile()
        f = h5py.File(path, 'w')
        self.h5 = f
        self.h5path = path
        self.tile_shape = None

    def add(self, key, val):
        raise NotImplementedError

    def update(self, key, val):
        raise NotImplementedError

    def reshape(self, targetshape):
        raise NotImplementedError

    def slice(self, slices):
        raise NotImplementedError

    def get(self, item):
        raise NotImplementedError

    def remove(self, key):
        raise NotImplementedError


class _tiles_h5_manager(_h5_manager):
    """
    Interface between tiles object and data management on disk by h5py.

    Args:
        slide_type (pathml.core.slide_types.SlideType): slide_type. must pass this if not loading from h5 (in which case
            slide_type can be also loaded from h5).
    """
    def __init__(self, h5=None, slide_type=None):
        super().__init__(h5 = h5)
        tilesgroup = self.h5.create_group('tiles')
        self.tiles = OrderedDict()
        if h5:
            for ds in h5.keys():
                if ds in ['array', 'masks', 'fields']:
                    h5.copy(ds, self.h5)
            # read tiles into RAM
            if 'tiles' in h5.keys(): 
                self.tiles = readtilesdicth5(h5['tiles']) 
                self.tile_shape = readtupleh5(h5['tiles'], 'tile_shape')

            if 'slide_type' in self.h5['fields'].keys():
                slidetype_dict = self.h5['fields']['slide_type']
                self.slide_type = pathml.core.SlideType(**slidetype_dict)
            else:
                self.slide_type = None
        else:
            fieldsgroup = self.h5.create_group('fields')
            self.slide_type = slide_type

    def add(self, tile):
        """
        Add tile to h5.

        Args:
            tile(pathml.core.tile.Tile): Tile object
        """
        if str(tile.coords) in self.tiles.keys():
           print(f"Tile is already in tiles. Overwriting {tile.coords} inplace.") 
        if self.tile_shape is None:
            self.tile_shape = tile.image.shape
        if tile.image.shape != self.tile_shape:
            raise ValueError(f"Tiles contains tiles of shape {self.tile_shape}, provided tile is of shape {tile.image.shape}"
                             f". We enforce that all Tile in Tiles must have matching shapes.")

        # check slide_type
        if not self.slide_type:
            self.slide_type = tile.slide_type
        else:
            if tile.slide_type != self.slide_type:
                raise ValueError(f"tile slide_type {tile.slide_type} does not match existing slide_type {self.slide_type}")

        if 'array' in self.h5.keys():
            # extend array if coords+shape is larger than self.h5['tiles/array'] 
            coords = tile.coords
            coordslength = len(coords)
            currentshape = self.h5['array'].shape[0:coordslength]
            requiredshape = [coord + tile_shape for coord, tile_shape in zip(coords, tile.image.shape[0:coordslength])]
            for dim, (current, required) in enumerate(zip(currentshape, requiredshape)):
                self.h5['array'].resize(required, axis=dim)

            # add tile to self.h5['tiles/array']
            slicer = [slice(coords[i], coords[i] + tile.image.shape[i]) for i in range(len(coords))] 
            self.h5['array'][tuple(slicer)] = tile.image

        # initialize self.h5['tiles/array'] if it does not exist 
        # note that the first tile is not necessarily (0,0) so we init with zero padding
        elif 'array' not in self.h5.keys():
            coords = list(tile.coords)
            shape = list(tile.image.shape)
            for dim, coord in enumerate(coords):
                shape[dim] += coord
            maxshape = tuple([None]*len(shape))
            self.h5.create_dataset(
                    'array', 
                    shape = shape,
                    maxshape = maxshape,
                    data = np.zeros(shape),
                    chunks = True,
                    compression = 'gzip',
                    compression_opts = 5,
                    shuffle = True
            )
            # write tile.image 
            slicer = [slice(coords[i], coords[i] + tile.image.shape[i]) for i in range(len(coords))] 
            self.h5['array'][tuple(slicer)] = tile.image
            # save tile shape as attribute to enforce consistency 
            self.tile_shape = tile.image.shape
            writetupleh5(self.h5['tiles'], 'tile_shape', tile.image.shape)

        if tile.masks:
            # create self.h5['tiles/masks']
            if 'masks' not in self.h5.keys():
                masksgroup = self.h5.create_group('masks')
                
            masklocation = tile.masks.h5manager.h5 if hasattr(tile.masks, 'h5manager') else tile.masks
            for mask in masklocation:
                # add masks to large mask array
                if mask in self.h5['masks'].keys():
                    # extend array 
                    coords = tile.coords
                    for i in range(len(coords)):
                        currentshape = self.h5['masks'][mask].shape[i]
                        requiredshape = coords[i] + tile.image.shape[i] 
                        if  currentshape < requiredshape:
                            self.h5['masks'][mask].resize(requiredshape, axis=i)
                    # add mask to mask array
                    slicer = [slice(coords[i], coords[i]+self.tile_shape[i]) for i in range(len(coords))] 
                    self.h5['masks'][mask][tuple(slicer)] = masklocation[mask][:]

                else:
                    # create mask array
                    maskarray = masklocation[mask][:]
                    coords = list(tile.coords)
                    coords = coords + [0]*len(maskarray.shape[len(coords):]) 
                    shape = [i + j for i,j in zip(maskarray.shape, coords)]
                    maxshape = tuple([None]*len(shape))
                    self.h5['masks'].create_dataset(
                            str(mask), 
                            shape = shape,
                            maxshape = maxshape,
                            data = np.zeros(shape),
                            chunks = True,
                            compression = 'gzip',
                            compression_opts = 5,
                            shuffle = True
                    )
                    # now overwrite by mask 
                    slicer = [slice(coords[i], coords[i] + tile.image.shape[i]) for i in range(len(coords))] 
                    self.h5['masks'][mask][tuple(slicer)] = maskarray 

        # add tile fields to tiles
        # note we don't put slide_type here: it should be the same for all tiles so we only store once at slide-level
        self.tiles[str(tile.coords)] = {
                'name': str(tile.name) if tile.name else None, 
                'labels': tile.labels if tile.labels else None,
                'coords': str(tile.coords)
        }

    def update(self, key, val, target):
        """
        Update a tile.

        Args:
            key(str): key of tile to be updated
            val(str): element that will replace target at key
            target(str): element of {all, image, labels} indicating field to be updated 
        """
        key = str(key)
        if key not in self.tiles.keys():
            raise ValueError(f"key {key} does not exist. Use add.")
        
        if target == 'all':
            assert self.tile_shape == val.image.shape, f"Cannot update a tile of shape {self.tile_shape} with a tile" \
                                                           f"of shape {val.image.shape}. Shapes must match."
            # overwrite
            self.add(val)
            print(f'tile at {key} overwritten')

        elif target == 'image':
            assert isinstance(val, np.ndarray), f"when replacing tile image must pass np.ndarray"
            assert self.tile_shape == val.shape, f"Cannot update a tile of shape {self.tile_shape} with a tile" \
                                                     f"of shape {val.shape}. Shapes must match."
            coords = list(eval(self.tiles[key]['coords']))
            slicer = [slice(coords[i], coords[i]+self.tile_shape[i]) for i in range(len(coords))] 
            self.h5['array'][tuple(slicer)] = val 
            print(f'array at {key} overwritten')

        elif target == 'masks':
            raise NotImplementedError

        elif target == 'labels':
            assert isinstance(val, (OrderedDict, dict)), f"when replacing labels must pass collections.OrderedDict of labels"
            self.tiles[key]['labels'] = val 
            print(f'label at {key} overwritten')

        else:
            raise KeyError('target must be all, image, masks, or labels')

    def get(self, item, slicer=None):
        """
        Retrieve tile from h5manager by key or index.

        Args:
            item(int, str, tuple): key or index of tile to be retrieved

        Returns:
            Tile(pathml.core.tile.Tile)
        """
        if isinstance(item, (str, tuple)):
            if str(item) not in self.tiles:
                raise KeyError(f'key {item} does not exist')
            tilemeta = self.tiles[str(item)]
        elif isinstance(item, int):
            if item > len(self.tiles) - 1:
                raise KeyError(f'index out of range, valid indices are ints in [0,{len(self.tiles) - 1}]')
            tilemeta = list(self.tiles.items())[item][1]
        else:
            raise KeyError(f'invalid item type: {type(item)}. must getitem by coord (type tuple[int]),'
                           f' index (type int), or name(type str)')
        # impute missing dimensions from self.tile_shape 
        coords = list(eval(tilemeta['coords']))
        if len(self.tile_shape) > len(coords):
            shape = list(self.tile_shape)
            coords = coords + [0]*len(shape[len(coords)-1:]) 
        tiler = [slice(coords[i], coords[i]+self.tile_shape[i]) for i in range(len(self.tile_shape))]
        tile = self.h5['array'][tuple(tiler)][:]

        # add masks to tile if there are masks
        if 'masks' in self.h5.keys():
            try:
                masks = {mask : self.h5['masks'][mask][tuple(tiler)][:] for mask in self.h5['masks']}
            except ValueError:
                # if mask is 2-d, need to only use first two dims of tiler
                masks = {mask: self.h5['masks'][mask][tuple(tiler)[0:2]][:] for mask in self.h5['masks']}
        else:
            masks = None

        if slicer:
            tile = tile[slicer]
            if masks is not None:
                masks = {key : masks[key][slicer] for key in masks}
        masks = pathml.core.masks.Masks(masks)

        return pathml.core.tile.Tile(tile, masks=masks, labels=tilemeta['labels'], name=tilemeta['name'],
                                     coords=eval(tilemeta['coords']), slide_type = self.slide_type)

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
        for key in self.tiles:
            yield self.get(key, slicer=slicer)
            
    def reshape(self, shape, centercrop = False):
        """
        Resample tiles to shape. 
        If shape does not evenly divide current tile shape, this method deletes tile labels and names.
        This method not mutate h5['tiles']['array'].

        Args:
            shape(tuple): new shape of tile.
            centercrop(bool): if shape does not evenly divide slide shape, take center crop
        """
        arrayshape = list(self.h5['array'].shape)
        # impute missing dimensions of shape from f['tiles/array'].shape 
        if len(arrayshape) > len(shape):
            shape = list(shape)
            shape = shape + arrayshape[len(shape):] 
        divisors = [range(int(n//d)) for n,d in zip(arrayshape, shape)]
        coordlist = list(itertools.product(*divisors))
        # multiply each element of coordlist by shape
        coordlist = [[int(c * s) for c,s in zip(coord, shape)] for coord in coordlist]
        if centercrop:
            offset = [int(n % d / 2) for n,d in zip(arrayshape, shape)]
            offsetcoordlist = []
            for item1, item2 in zip(offset, coordlist):
                offsetcoordlist.append(tuple([int(item1 + x) for x in item2])) 
            coordlist = offsetcoordlist
        newtilesdict = OrderedDict()
        # if shape evenly divides arrayshape transfer labels
        remainders = [int(n % d) for n,d in zip(arrayshape, shape)]
        offsetstooriginal = [[int(n % d) for n,d in zip(coord, self.tile_shape)] for coord in coordlist] 
        if all(x<=y for x, y in zip(shape, arrayshape)) and all(rem == 0 for rem in remainders):
            # transfer labels
            for coord, off in zip(coordlist, offsetstooriginal):
                # find coordinate from which to transfer labels 
                oldtilecoordlen = len(eval(list(self.tiles.keys())[0]))
                oldtilecoord = [int(x-y) for x,y in zip(coord, off)]
                oldtilecoord = oldtilecoord[:oldtilecoordlen]
                labels = self.tiles[str(tuple(oldtilecoord))]['labels']
                name = self.tiles[str(tuple(oldtilecoord))]['name']
                newtilesdict[str(tuple(coord))] = {
                        'name': None, 
                        'labels': labels,
                        'coords': str(tuple(coord)), 
                        'slidetype': None
                }
        else: 
            # TODO: fix tests (monkeypatch) to implement the check above (the y/n hangs)
            '''
            choice = None
            yes = {'yes', 'y'}
            no = {'no', 'n'}
            while choice not in yes or no:
                choice = input('Reshaping to a shape that does not evenly divide old tile shape deletes labels and names. Would you like to continue? [y/n]\n').lower()
                if choice in yes: 
                    for coord in coordlist:
                        newtilesdict[str(tuple(coord))] = {
                                'name': None, 
                                'labels': None,
                                'coords': str(tuple(coord)), 
                                'slidetype': None
                        }
                elif choice in no:
                    raise Exception(f"User cancellation.")
                else:
                    sys.stdout.write("Please respond with 'y' or 'n'")
            '''
            for coord in coordlist:
                newtilesdict[str(tuple(coord))] = {
                        'name': None, 
                        'labels': None,
                        'coords': str(tuple(coord)), 
                        'slidetype': None
                }
        self.tiles = newtilesdict
        self.tile_shape = shape

    def remove(self, key):
        """
        Remove tile from self.h5 by key.
        """
        if not isinstance(key, (str, tuple)):
            raise KeyError(f'key must be str or tuple, check valid keys in repr')
        if str(key) not in self.tiles:
            raise KeyError(f'key {key} is not in Tiles')
        del self.tiles[str(key)]


class _masks_h5_manager(_h5_manager):
    """
    Interface between masks object and data management on disk by h5py. 
    """
    def __init__(self, h5 = None):
        super().__init__(h5 = h5)
        if h5:
            for ds in h5.keys():
                h5.copy(ds, self.h5)

    def add(self, key, mask):
        """
        Add mask to h5.

        Args:
            key(str): key labeling mask
            mask(np.ndarray): mask array  
        """
        if not isinstance(mask, np.ndarray):
            raise ValueError(f"can not add {type(mask)}, mask must be of type np.ndarray")
        if not isinstance(key, str):
            raise ValueError(f"invalid type {type(key)}, key must be of type str")
        if key in self.h5.keys():
            raise ValueError(f"key {key} already exists. Cannot add. Must update to modify existing mask.")
        if self.tile_shape is None:
            self.tile_shape = mask.shape
        if mask.shape != self.tile_shape:
            raise ValueError(
                f"Masks contains masks of shape {self.tile_shape}, provided mask is of shape {mask.shape}. "
                f"We enforce that all Mask in Masks must have matching shapes.")
        newkey = self.h5.create_dataset(
            bytes(str(key), encoding = 'utf-8'),
            data = mask
        )

    def update(self, key, mask):
        """
        Update a mask.

        Args:
            key(str): key indicating mask to be updated
            mask(np.ndarray): mask
        """
        if key not in self.h5.keys():
            raise ValueError(f"key {key} does not exist. Must use add.")

        original_mask = self.get(key)

        assert original_mask.shape == mask.shape, f"Cannot update a mask of shape {original_mask.shape} with a mask" \
                                                  f"of shape {mask.shape}. Shapes must match."

        self.h5[key][...] = mask

    def slice(self, slicer):
        """
        Generator slicing all tiles, extending numpy array slicing.

        Args:
            slicer: List where each element is an object of type slice https://docs.python.org/3/c-api/slice.html
                    indicating how the corresponding dimension should be sliced. The list length should correspond to the
                    dimension of the tile. For 2D H&E images, pass a length 2 list of slice objects.
        Yields:
            key(str): mask key
            val(np.ndarray): mask
        """
        for key in self.h5.keys():
            yield key, self.get(key, slicer=slicer) 

    def get(self, item, slicer=None):
        # must check bool separately, since isinstance(True, int) --> True
        if isinstance(item, bool) or not (isinstance(item, str) or isinstance(item, int)):
            raise KeyError(f"key of type {type(item)} must be of type str or int")

        if isinstance(item, str):
            if item not in self.h5.keys():
                raise KeyError(f'key {item} does not exist')
            if slicer is None:
                return self.h5[item][:]
            return self.h5[item][:][tuple(slicer)]

        else:
            if item > len(self.h5) - 1:
                raise KeyError(f"index out of range, valid indices are ints in [0,{len(self.h5['masks'].keys()) - 1}]")
            if slicer is None:
                return self.h5[list(self.h5.keys())[item]][:]
            return self.h5[list(self.h5.keys())[item]][:][tuple(slicer)]

    def remove(self, key):
        """
        Remove mask by key.

        Args:
            key(str): key indicating mask to be removed
        """
        if not isinstance(key, str):
            raise KeyError(f"masks keys must be of type(str) but key was passed of type {type(key)}")
        if key not in self.h5.keys():
            raise KeyError('key is not in Masks')
        del self.h5[key]
