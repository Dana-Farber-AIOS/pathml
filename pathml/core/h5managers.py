import h5py
import tempfile
import ast
from collections import OrderedDict
import numpy as np
import itertools

from pathml.core.utils import writedataframeh5, writestringh5, writedicth5, writetupleh5, readtupleh5
import pathml.core.masks
import pathml.core.tile

"""
h5: 
*fields
    slide_backend
    name
    labels
    history
*array
*masks
   mask1
   mask2
   ...

put tilesdict in h5 when write, remove when read
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
        self.tilesdict = OrderedDict()
        tilesgroup = self.h5.create_group('tiles')
        if h5:
            for ds in h5.keys():
                h5.copy(ds, f)
            self.tilesdict = dict(f['tiles'].attrs['tilesdict'].astype(str)) if 'tilesdict' in f['tiles'].attrs.keys() else None 
            if self.tilesdict:
                del f['tiles'].attrs['tilesdict']
                # TODO: coerce each element in tilesdict to correct types 
                for key in self.tilesdict.keys()
                    k = self.tilesdict[key]
                    k['labels'] = dict(k['labels']) 
                    k['labels'] = {k:v for k,v in k['labels'].items()}
                    k['coords'] = eval(k['coords']) 
            self.shape = readtupleh5(f['tiles/array'], 'shape')

        self.tilesdict[tile.coords] = {
                'name': tile.name, 
                'labels': tile.labels,
                'coords': tile.coords, 
                'slidetype': tile.slidetype
        }

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
           print(f"Tile is already in tiles. Overwriting {key} inplace.") 
        if self.shape is None:
            self.shape = tile.image.shape
        if tile.image.shape != self.shape:
            raise ValueError(f"Tiles contains tiles of shape {self.shape}, provided tile is of shape {tile.image.shape}"
                             f". We enforce that all Tile in Tiles must have matching shapes.")

        if 'array' in self.h5.keys():
            # extend array 
            coords = tile.coords
            for i in len(coords):
                # TODO: off-by-one error?
                currentshape = self.h5['tiles/array'].shape[i]
                requiredshape = coords[i] + self.shape[i] - 1
                if  currentshape < requiredshape:
                    self.h5['tiles/array'].resize(self.h5['tiles/array'].shape[i]+self.shape[i], axis=i)
                    self.h5['tiles/array'].resize(self.h5['tiles/array'].shape[i] + (currentshape - requiredshape), axis=i)
            # add tile to array
            slicer = [slice(coords[i], coords[i]+self.shape[i]) for i in len(coords)] 
            self.h5['tiles/array'][tuple(slicer)] = tile.image

        # TODO: just 'else'?
        elif 'array' not in self.h5['tiles'].keys():
            maxshape = tuple([None]*len(self.shape))
            self.h5['tiles'].create_dataset(
                    'array', 
                    shape = self.shape,
                    maxshape = maxshape,
                    data = tile.image,
                    chunks = True,
                    compression = 'gzip',
                    compression_opts = 5,
                    shuffle = True
            )
            self.h5['tiles/array'].attr['shape'] = tile.image.shape

        if tile.masks:
            if 'masks' not in self.h5.keys():
                masksgroup = self.h5.create_group('masks')
                
            masklocation = tile.masks.h5manager.h5 if hasattr(tile.masks, 'h5manager') else tile.masks

            for mask in masklocation:
                if mask in self.h5['tiles/masks'].keys():
                    # extend array 
                    coords = tile.coords
                    for i in range(len(coords)):
                        # TODO: off-by-one error?
                        if self.h5['tiles/masks'][mask].shape[i] < coords[i] + self.shape[i] - 1:
                            self.h5['tiles/masks'][mask].resize(self.h5['tiles/masks'][mask].shape[i]+self.shape[i], axis=i)
                    # add mask to array
                    slicer = [slice(coords[i], coords[i]+self.shape[i]) for i in range(len(coords))] 
                    self.h5['tiles/array'][tuple(slicer)] = masklocation[mask][:]

                # create mask array
                else:
                    maskarray = masklocation[mask][:]
                    maxshape = tuple([None]*len(self.shape))
                    self.h5['tiles/masks'].create_dataset(
                            str(mask), 
                            shape = self.shape,
                            maxshape = maxshape,
                            data = maskarray,
                            chunks = True,
                            compression = 'gzip',
                            compression_opts = 5,
                            shuffle = True
                    )
        # hold dict with tile information in RAM
        # TODO: coords redundant?
        self.tilesdict[tile.coords] = {
                'name': tile.name, 
                'labels': tile.labels,
                'coords': tile.coords, 
                'slidetype': tile.slidetype
        }

    def update(self, key, val, target):
        key = str(key)
        if key not in self.tilesdictkeys():
            raise ValueError(f"key {key} does not exist. Use add.")
        
        if target == 'all':
            # TODO: check 
            # assert isinstance(val, Tile), f"when replacing whole tile, must pass a Tile object"
            assert self.shape == val.image.shape, f"Cannot update a tile of shape {self.shape} with a tile" \
                                                           f"of shape {val.image.shape}. Shapes must match."
            # overwrite
            self.add(key, val)
            print(f'tile at {key} overwritten')

        elif target == 'image':
            assert isinstance(val, np.ndarray), f"when replacing tile image must pass np.ndarray"
            assert self.shape == val.shape, f"Cannot update a tile of shape {self.shape} with a tile" \
                                                     f"of shape {val.shape}. Shapes must match."
            # get coords
            # add to self.h5['array'] at coords
            coords = list(self.tilesdict[key]['coords'])
            slicer = [slice(coords[i], coords[i]+self.shape[i]) for i in len(coords)] 
            self.h5['tiles/array'][tuple(slicer)] = val 
            print(f'image at {key} overwritten')

        elif target == 'masks':
            raise NotImplementedError

        elif target == 'labels':
            assert isinstance(val, (OrderedDict, dict)), f"when replacing labels must pass collections.OrderedDict of labels"
            self.tilesdict[key]['labels'] = val 
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
        if not isinstance(item, (int, str, tuple)):
            raise KeyError(f'must getitem by coordinate(type tuple[int]), index(type int), or name(type str)')
        if isinstance(item, (str, tuple)):
            if item not in self.tilesdict:
                raise KeyError(f'key {item} does not exist')
        if isinstance(item, int):
            if item > len(self.tilesdict) - 1:
                raise KeyError(f'index out of range, valid indices are ints in [0,{len(self.tilesdict) - 1}]')
        # since tilesdict is OrderedDict both int and tuple/str work
        tilemeta = self.tilesdict[item]
        # extend missing dimensions of coords by imputing from self.shape 
        coords = list(tilemeta['coords'])
        if len(self.shape) > len(coords):
            shape = list(self.shape)
            coords = coords + shape[len(coords)-1:] 

        tiler = [slice(coords[i], coords[i]+self.shape[i]) for i in range(len(self.shape))]
        tile = self.h5['tiles/array'][tiler][:]
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
        for key in self.tilesdict:
            yield self.get(key, slices=slices)
            
    def reshape(self, shape, centercrop = False):
        """
        Resample tiles to new shape. 
        This method deletes tile labels and names.
        This method does not delete any pixels from the slide array, to restore the full image choose a shape that evenly divides slide shape.

        Args:
            shape(tuple): new shape of tile.
            centercrop(bool): if shape does not evenly divide slide shape, take center crop
        """
        arrayshape = list(f['array'].shape)
        # extend missing dimensions of shape by imputing from f['array'].shape 
        if len(arrayshape) > len(shape):
            shape = list(shape)
            shape = shape + arrayshape[len(shape)-1:] 
        divisors = [range(n//d) for n,d in zip(arrayshape, shape)]
        coordlist = itertools.product(*divisors)
        slidetype = self.tilesdict[0]['slidetype']
        self.tilesdict = OrderedDict()
        for coord in coordlist:
            self.tilesdict[coords] = {
                    'name': None, 
                    'labels': None,
                    'coords': coords, 
                    'slidetype': slidetype
            }

    def remove(self, key):
        """
        Remove tile from self.h5 by key.
        """
        if not isinstance(key, (str, tuple)):
            raise KeyError(f'key must be str or tuple, check valid keys in repr')
        if str(key) not in self.tilesdict:
            raise KeyError(f'key {key} is not in Tiles')
        del self.tilesdict[str(key)]


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
