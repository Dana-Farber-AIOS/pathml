import h5py
import tempfile
import ast
from collections import OrderedDict
import numpy as np

from pathml.core.utils import writedataframeh5, writestringh5, writedicth5, writetupleh5
import pathml.core.masks
import pathml.core.tile

"""
h5manager
    tiledict
    h5

h5 object: 
*fields
*array
*masks
   mask1
   mask2
   ...

put tiledict in h5 when write, remove when read
"""


class _h5_manager:
    """
    Abstract class for h5 data management
    """

    def __init__(self, h5 = None):
        path = tempfile.TemporaryFile()
        f = h5py.File(path, 'w')
        self.h5 = f
        self.h5path = path
        self.shape = None
        if h5:
            for ds in h5.keys():
                h5.copy(ds, f)
 

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
    """

    def __init__(self, h5 = None):
        super().__init__(h5 = h5)

    def add(self, key, tile):
        """
        Add tile to self.h5 as dataset indexed by key with metadata as attributes.

        Args:
            key(str or tuple): key will become tile name 
            tile(pathml.core.tile.Tile): Tile object
        """
        # what to do with names? need to replace with coordinates
        # when add a tile
        #   if array doesn't exist 
        #       init array self.h5['tiles']['array']
        #       init attribute holding size self.h5['tiles']['array'].attrs['tileshape'] 
        #       put name and coordinates in dict stored in slidedata 
        #           at least not stored in h5 because we don't want to need read twice
        #   if array exists
        #       extend by size
        #       add tile

        if not isinstance(key, (str, tuple)):
            raise ValueError(f"can not add type {type(key)}, key must be of type str or tuple")
        if str(key) in self.h5.keys():
           raise KeyError(f"Tile is already in tiles. Call remove or replace.") 
        if self.shape is None:
            self.shape = tile.image.shape
        if tile.image.shape != self.shape:
            raise ValueError(f"Tiles contains tiles of shape {self.shape}, provided tile is of shape {tile.image.shape}"
                             f". We enforce that all Tile in Tiles must have matching shapes.")

        tilegroup = self.h5.create_group(str(key))
        masksgroup = tilegroup.create_group('masks')
        writedataframeh5(tilegroup, 'tile', tile.image)

        if tile.masks:
            try:
                for mask in tile.masks.h5manager.h5:
                    writedataframeh5(masksgroup, str(mask), tile.masks.h5manager.h5[mask][:])
            except:
                for mask in tile.masks:
                    writedataframeh5(masksgroup, str(mask), tile.masks[mask])

        if tile.labels:
            writedicth5(tilegroup, 'labels', tile.labels)

        if tile.coords:
            writetupleh5(tilegroup, 'coords', tile.coords)

        if tile.slidetype:
            writestringh5(tilegroup, 'slidetype', tile.slidetype)

        if tile.name:
            writestringh5(tilegroup, 'name', tile.name)

    def update(self, key, val, target):
        key = str(key)
        if key not in self.h5.keys():
            raise ValueError(f"key {key} does not exist. Use add.")
         
        tile = self.get(key)
        original_tile = tile.image
        
        if target == 'all':
            #TODO: check somewhere
            # assert isinstance(val, Tile), f"when replacing whole tile, must pass a Tile object"
            assert original_tile.shape == val.image.shape, f"Cannot update a tile of shape {original_tile.shape} with a tile" \
                                                           f"of shape {val.image.shape}. Shapes must match."
            self.remove(key)
            self.add(key, val)

        elif target == 'image':
            assert isinstance(val, np.ndarray), f"when replacing tile image must pass np.ndarray"
            assert original_tile.shape == val.shape, f"Cannot update a tile of shape {original_tile.shape} with a tile" \
                                                     f"of shape {val.shape}. Shapes must match."
            self.h5[key]['tile'][...] = val

        elif target == 'masks':
            raise NotImplementedError

        elif target == 'labels':
            assert isinstance(val, (OrderedDict, dict)), f"when replacing labels must pass collections.OrderedDict of labels"
            labelarray = np.array(list(val.items()), dtype=object)
            self.h5[key].attrs['labels'] = labelarray

        else:
            raise KeyError('target must be all, image, masks, or labels')

    def get(self, item, slices=None):
        if isinstance(item, (str, tuple)):
            if str(item) not in self.h5.keys():
                raise KeyError(f'key {item} does not exist')
            # str|tuple key, no slicing
            if slices is None:
                k = self.h5[str(item)]
                tile = k['tile'][:]
                maskdict = {key:k['masks'][key][...] for key in k['masks'].keys()} if 'masks' in k.keys() else None 
                name = k.attrs['name'] if 'name' in k.attrs.keys() else None
                labels = dict(k.attrs['labels'].astype(str)) if 'labels' in k.attrs.keys() else None
                coords = eval(k.attrs['coords']) if 'coords' in k.attrs.keys() else None
                slidetype = k.attrs['slidetype'] if 'slidetype' in k.attrs.keys() else None
                return pathml.core.tile.Tile(tile, masks=pathml.core.masks.Masks(maskdict), labels=labels, name=str(name), coords=coords, slidetype=slidetype)

            # str|tuple key, with slicing
            k = self.h5[str(item)]
            tile = k['tile'][tuple(slices)]
            maskdict = {key:k['masks'][key][tuple(slices)] for key in k['masks'].keys()} if 'masks' in k.keys() else None 
            name = k.attrs['name'] if 'name' in k.attrs.keys() else None
            labels = dict(k.attrs['labels'].astype(str)) if 'labels' in k.attrs.keys() else None
            coords = eval(k.attrs['coords']) if 'coords' in k.attrs.keys() else None
            slidetype = k.attrs['slidetype'] if 'slidetype' in k.attrs.keys() else None
            return pathml.core.tile.Tile(tile, masks=pathml.core.masks.Masks(maskdict), labels=labels, name=str(name), coords=coords, slidetype=slidetype)
        
        if not isinstance(item, int):
            raise KeyError(f"must getitem by coordinate(type tuple[int]) or index(type int)")
        if item > len(self.h5) - 1:
            raise KeyError(f"index out of range, valid indices are ints in [0,{len(self.h5) - 1}]")
        # int key, no slicing
        if slices is None:
            k = self.h5[list(self.h5.keys())[item]]
            tile = k['tile'][:]
            maskdict = {key: k['masks'][key][...] for key in k['masks'].keys()} if 'masks' in k.keys() else None
            name = k.attrs['name'] if 'name' in k.attrs.keys() else None
            labels = k.attrs['labels'] if 'labels' in k.attrs.keys() else None
            coords = eval(k.attrs['coords']) if 'coords' in k.attrs.keys() else None
            slidetype = k.attrs['slidetype'] if 'slidetype' in k.attrs.keys() else None
            return pathml.core.tile.Tile(tile, masks=pathml.core.masks.Masks(maskdict), labels=labels, name=str(name), coords=coords, slidetype=slidetype)

        # TODO: move slicing logic
        # int key, with slicing
        k = self.h5[list(self.h5.keys())[item]]
        tile = k['tile'][tuple(slices)]
        maskdict = {key: k['masks'][key][tuple(slices)] for key in k['masks'].keys()} if 'masks' in k.keys() else None
        name = k.attrs['name'] if 'name' in k.attrs.keys() else None
        labels = k.attrs['labels'] if 'labels' in k.attrs.keys() else None
        coords = eval(k.attrs['coords']) if 'coords' in k.attrs.keys() else None
        slidetype = k.attrs['slidetype'] if 'slidetype' in k.attrs.keys() else None
        return pathml.core.tile.Tile(tile, masks=pathml.core.masks.Masks(maskdict), labels=labels, name=str(name), coords=coords, slidetype=slidetype)

    def slice(self, slices):
        """
        Generator to slice all tiles in self.h5 extending numpy array slicing

        Args:
            slices: List where each element is an object of type slice https://docs.python.org/3/c-api/slice.html
                    indicating how the corresponding dimension should be sliced. The list length should correspond to the
                    dimension of the tile. For 2D H&E images, pass a  length 2 list of slice objects.

        Yields:
            key(str): tile coordinates
            val(pathml.core.tile.Tile): tile
        """
        for key in self.h5.keys():
            yield self.get(key, slices=slices)
            
    def reshape(self, shape):
        """
        Resample tiles to new shape. 

        Args:
            shape: new shape of tile.

        (support change inplace and return copy) 
        """
        raise NotImplementedError

    def remove(self, key):
        """
        Remove tile from self.h5 by key.
        """
        if not isinstance(key, (str, tuple)):
            raise KeyError(f'key must be str or tuple, check valid keys in repr')
        if str(key) not in self.h5.keys():
            raise KeyError(f'key {key} is not in Tiles')
        del self.h5[str(key)]


class _masks_h5_manager(_h5_manager):
    """
    Interface between masks object and data management on disk by h5py. 
    """

    def __init__(self, h5 = None):
        super().__init__(h5 = h5)

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
        if key in self.h5.keys():
            raise ValueError(f"key {key} already exists. Cannot add. Must update to modify existing mask.")
        if self.shape is None:
            self.shape = mask.shape
        if mask.shape != self.shape:
            raise ValueError(
                f"Masks contains masks of shape {self.shape}, provided mask is of shape {mask.shape}. "
                f"We enforce that all Mask in Masks must have matching shapes.")
        newkey = self.h5.create_dataset(
            bytes(str(key), encoding = 'utf-8'),
            data = mask
        )

    def update(self, key, mask):
        """
        Update an existing mask.

        Args:
            key(str): key labeling mask
            mask(np.ndarray): mask
        """
        if key not in self.h5.keys():
            raise ValueError(f"key {key} does not exist. Must use add.")

        original_mask = self.get(key)

        assert original_mask.shape == mask.shape, f"Cannot update a mask of shape {original_mask.shape} with a mask" \
                                                  f"of shape {mask.shape}. Shapes must match."

        self.h5[key][...] = mask

    def slice(self, slices):
        """
        Generator to slice all masks in self.h5 extending numpy array slicing.

        Args:
            slices: list where each element is an object of type slice https://docs.python.org/3/c-api/slice.html
                    indicating how the corresponding dimension should be sliced
        Yields:
            key(str): mask key
            val(np.ndarray): mask
        """
        for key in self.h5.keys():
            yield key, self.get(key, slices=slices) 

    def reshape(self, targetshape):
        pass

    def get(self, item, slices=None):
        # check type of input
        # must check bool separately, since isinstance(True, int) --> True
        if isinstance(item, bool) or not (isinstance(item, str) or isinstance(item, int)):
            raise KeyError(f"key of type {type(item)} must be of type str or int")

        if isinstance(item, str):
            if item not in self.h5.keys():
                raise KeyError(f'key {item} does not exist')
            if slices is None:
                return self.h5[item][:]
            return self.h5[item][tuple(slices)]

        else:
            if item > len(self.h5) - 1:
                raise KeyError(f"index out of range, valid indices are ints in [0,{len(self.h5['masks'].keys()) - 1}]")
            if slices is None:
                return self.h5[list(self.h5.keys())[item]][:]
            return self.h5[list(self.h5.keys())[item]][tuple(slices)]

    def remove(self, key):
        """
        Remove mask from self.h5 by key.
        """
        if not isinstance(key, str):
            raise KeyError(f"masks keys must be of type(str) but key was passed of type {type(key)}")
        if key not in self.h5.keys():
            raise KeyError('key is not in Masks')
        del self.h5[key]
