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
        OrderedDict (iterate over index)
            {coords : {name,labels,etc}} 
    h5

h5: 
*fields
    backend
    labels
    history
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
        self.tiledict = OrderedDict()
        if h5:
            # TODO: read tiledict and del from h5
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

        if not isinstance(key, (str, tuple)):
            raise ValueError(f"can not add type {type(key)}, key must be of type str or tuple")
        if str(key) in self.h5.keys():
           raise KeyError(f"Tile is already in tiles. Call remove or replace.") 
        if self.shape is None:
            self.shape = tile.image.shape
        if tile.image.shape != self.shape:
            raise ValueError(f"Tiles contains tiles of shape {self.shape}, provided tile is of shape {tile.image.shape}"
                             f". We enforce that all Tile in Tiles must have matching shapes.")

        if 'array' in self.h5.keys():
            # extend array 
            # TODO: Jacob: what if tiles are not contiguous? will need to grow the array including empty tiles
            coords = tile.coords
            for i in len(coords):
                # TODO: off-by-one error?
                if self.h5['array'].shape[i] < coords[i] + self.shape[i] - 1:
                    self.h5['array'].resize(self.h5['array'].shape[i]+self.shape[i], axis=i)
            # add tile to array
            slicer = [slice(coords[i], coords[i]+self.shape[i]) for i in len(coords)] 
            self.h5['array'][tuple(slicer)] = tile.image

        elif 'array' not in self.h5.keys():
            maxshape = tuple([None]*len(self.shape))
            self.h5.create_dataset(
                    'array', 
                    shape = self.shape,
                    maxshape = maxshape,
                    data = tile.image,
                    chunks = True,
                    compression = 'gzip',
                    compression_opts = 5,
                    shuffle = True
            )
            h5['array'].attr['shape'] = tile.image.shape

        if tile.masks:
            if 'masks' not in self.h5.keys():
                masksgroup = self.h5.create_group('masks')
                
            masklocation = tile.masks.h5manager.h5 if hasattr(tile.masks, 'h5manager') else tile.masks

            for mask in masklocation:
                if mask in self.h5['masks'].keys():
                    # extend array 
                    coords = tile.coords
                    for i in range(len(coords)):
                        # TODO: off-by-one error?
                        if self.h5['masks'][mask].shape[i] < coords[i] + self.shape[i] - 1:
                            self.h5['masks'][mask].resize(self.h5['masks'][mask].shape[i]+self.shape[i], axis=i)
                    # add mask to array
                    slicer = [slice(coords[i], coords[i]+self.shape[i]) for i in range(len(coords))] 
                    self.h5['array'][tuple(slicer)] = masklocation[mask][:]

                # create mask array
                else:
                    maskarray = masklocation[mask][:]
                    maxshape = tuple([None]*len(self.shape))
                    self.h5['masks'].create_dataset(
                            str(mask), 
                            shape = self.shape,
                            maxshape = maxshape,
                            data = maskarray,
                            chunks = True,
                            compression = 'gzip',
                            compression_opts = 5,
                            shuffle = True
                    )
        # save tile metadata indexed by coords
        # TODO: coords redundant?
        self.tiledict[tile.coords] = {
                'name': tile.name, 
                'labels': tile.labels,
                'coords': tile.coords, 
                'slidetype': tile.slidetype
        }

    def update(self, key, val, target):
        key = str(key)
        if key not in self.h5.keys():
            raise ValueError(f"key {key} does not exist. Use add.")
         
        tile = self.get(key)
        original_tile = tile.image
        
        if target == 'all':
            # TODO: check 
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

    def get(self, item, slicer=None):
        if not isinstance(item, (int, str, tuple)):
            raise KeyError(f'must getitem by coordinate(type tuple[int]), index(type int), or name(type str)')
        if isinstance(item, (str, tuple)):
            if item not in self.tiledict:
                raise KeyError(f'key {item} does not exist')
        if isinstance(item, int):
            if item > len(self.tiledict) - 1:
                raise KeyError(f'index out of range, valid indices are ints in [0,{len(self.tiledict) - 1}]')
        # since tiledict is OrderedDict both int and tuple/str work
        tilemeta = self.tiledict[item]
        # TODO: handle # channels, coords currently is just (x,y) we need (x,y,z,c,t)
        # tilemeta['coords'] for channel should be 0
        # if slidetype is HESlide infer coord c=0 then cut c=0-c=2
        tiler = [slice(tilemeta['coords'][i], tilemeta['coords'][i]+self.shape[i]) for i in range(len(self.shape))]
        tile = self.h5['array'][tiler][:]
        masks = pathml.core.masks.Masks({self.h5['masks'][mask][tiler][:] for mask in self.h5['masks']})
        if slicer:
            tile = tile[slicer]
            masks = {mask[slicer] for mask in masks}

        return pathml.core.tile.Tile(tile, masks=masks, labels=tilemeta['labels'], name=tilemeta['name'], coords=tilemeta['coords'], slidetype=tilemeta['slidetype'])

    def slice(self, slicer):
        """
        Generator to slice all tiles in self.h5 extending numpy array slicing

        Args:
            slicer: List where each element is an object of type slice https://docs.python.org/3/c-api/slice.html
                    indicating how the corresponding dimension should be sliced. The list length should correspond to the
                    dimension of the tile. For 2D H&E images, pass a  length 2 list of slice objects.

        Yields:
            key(str): tile coordinates
            val(pathml.core.tile.Tile): tile
        """
        for key in self.tiledict:
            yield self.get(key, slices=slices)
            
    def reshape(self, shape):
        """
        Resample tiles to new shape. 

        Args:
            shape: new shape of tile.
        """
        # this is simply reindexing self.h5['array']
        # check shape of large array
        # find coords which are valid cuts of array
        # repopulate self.tiledict from valid cuts

    # TODO: remove doesn't make sense in the context of new framework
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
