"""
Copyright 2021, Dana-Farber Cancer Institute and Weill Cornell Medicine
License: GNU GPL 2.0
"""

import h5py
import tempfile
from collections import OrderedDict
import numpy as np
import itertools
import anndata
import os

import pathml.core.masks
import pathml.core.tile
import pathml.core
from pathml.core.utils import writetupleh5, readtupleh5, readtilesdicth5, readcounts

class h5pathManager():
    """
    Interface between slidedata object and data management on disk by h5py.
    """
    def __init__(self, h5path = None, slidedata = None):
        path = tempfile.TemporaryFile()
        f = h5py.File(path, 'w')
        self.h5 = f
        # keep a reference to h5 tempfile so that it is never garbage collected
        self.h5reference = path
        # create temporary directory for slidedata.counts
        countspath = tempfile.TemporaryDirectory()
        self.countspath = path
        self.counts = anndata.AnnData()
        if h5path:
            assert not slidedata, f"if creating h5pathmanager from h5path, slidedata should not be required"
            # TODO: implement isvalidh5path
            assert check_valid_h5path_format(h5path), f"h5path must conform to .h5path standard, see documentation"
            # copy h5path into self.h5 
            for ds in h5path.keys():
                if ds in ['fields', 'array', 'masks', 'tiles']:
                    h5path.copy(ds, self.h5)
                if ds in ['counts']:
                    # TODO: clean readcounts
                    self.counts = readcounts(h5path['counts'])
                    with self.countspath as tmpdirname:
                        self.counts.filename = tmpdirname + '/tmpfile.h5ad'
            
        else:
            assert slidedata, f"must pass slidedata object to create h5path"
            # fields
            #    create group
            fieldsgroup = self.h5.create_group("fields")
            fieldsgroup.attrs['name'] = slidedata.name 
            labelsgroup = self.h5["fields"].create_group("labels")
            if slidedata.labels:
                for key, label in slidedata.labels.items():
                    self.h5["fields/labels"].attrs[key] = label
            slidetypegroup = self.h5["fields"].create_group("slide_type")
            # TODO: implement slide_type asdict method
            if slidedata.slide_type:
                for key, val in slidedata.slide_type.asdict().items():
                    self.h5["fields/slide_type"].attrs[key] = val
            # tiles
            tilesgroup = self.h5.create_group("tiles")
            tilesgroup.attrs['tile_shape'] = 0
            # array
            self.h5.create_dataset("array", data=h5py.Empty("f"))
            # masks
            masksgroup = self.h5.create_group("masks")
            # counts
            countsgroup = self.h5.create_group("counts")
        
        slide_type_dict = {key:val for key, val in self.h5["fields/slide_type"].items()}
        self.slide_type = pathml.core.slide_types.SlideType(**slide_type_dict)

    def add_tile(self, tile):
        """
        Add tile to h5.

        Args:
            tile(pathml.core.tile.Tile): Tile object
        """
        if str(tile.coords) in self.tiles.keys():
           print(f"Tile is already in tiles. Overwriting {tile.coords} inplace.") 
           # remove old cells from self.counts so they do not duplicate
           if tile.counts:
               if "tile" in self.counts.obs.keys():
                   self.counts = self.counts[self.counts.obs['tile'] != tile.coords]
        if self.tile_shape is None:
            self.tile_shape = tile.image.shape
        if tile.image.shape != self.tile_shape:
            raise ValueError(f"Tiles contains tiles of shape {self.tile_shape}, provided tile is of shape {tile.image.shape}"
                             f". We enforce that all Tile in Tiles must have matching shapes.")

        # check slide_type
        if tile.slide_type != self.slide_type:
            raise ValueError(f"tile slide_type {tile.slide_type} does not match existing slide_type {self.slide_type}")

        if self.h5["array"].shape:
            # extend array if coords+shape is larger than self.h5['tiles/array'] 
            coords = tile.coords
            coordslength = len(coords)
            currentshape = self.h5["array"].shape[0:coordslength]
            requiredshape = [coord + tile_shape for coord, tile_shape in zip(coords, tile.image.shape[0:coordslength])]
            for dim, (current, required) in enumerate(zip(currentshape, requiredshape)):
                self.h5["array"].resize(required, axis=dim)

            # add tile to self.h5["array"]
            slicer = [slice(coords[i], coords[i] + tile.image.shape[i]) for i in range(len(coords))] 
            self.h5["array"][tuple(slicer)] = tile.image

        # initialize self.h5['array'] if it does not exist
        # note that the first tile is not necessarily (0,0) so we init with zero padding
        else:
            coords = list(tile.coords)
            shape = list(tile.image.shape)
            for dim, coord in enumerate(coords):
                shape[dim] += coord
            maxshape = tuple([None]*len(shape))
            self.h5.create_dataset(
                    "array", 
                    shape = shape,
                    maxshape = maxshape,
                    data = np.zeros(shape),
                    chunks = True,
                    compression = "gzip",
                    compression_opts = 5,
                    shuffle = True
            )
            # write tile.image 
            slicer = [slice(coords[i], coords[i] + tile.image.shape[i]) for i in range(len(coords))] 
            self.h5["array"][tuple(slicer)] = tile.image
            # save tile shape as attribute to enforce consistency 
            self.tile_shape = tile.image.shape
            writetupleh5(self.h5["tiles"], "tile_shape", tile.image.shape)

        if tile.masks:
            # create self.h5["masks"]
            if "masks" not in self.h5.keys():
                masksgroup = self.h5.create_group("masks")
                
            masklocation = tile.masks.h5manager.h5 if hasattr(tile.masks, 'h5manager') else tile.masks
            for mask in masklocation:
                # add masks to large mask array
                if mask in self.h5["masks"].keys():
                    # extend array 
                    coords = tile.coords
                    for i in range(len(coords)):
                        currentshape = self.h5["masks"][mask].shape[i]
                        requiredshape = coords[i] + tile.image.shape[i] 
                        if  currentshape < requiredshape:
                            self.h5["masks"][mask].resize(requiredshape, axis=i)
                    # add mask to mask array
                    slicer = [slice(coords[i], coords[i]+self.tile_shape[i]) for i in range(len(coords))] 
                    self.h5["masks"][mask][tuple(slicer)] = masklocation[mask][:]

                else:
                    # create mask array
                    maskarray = masklocation[mask][:]
                    coords = list(tile.coords)
                    coords = coords + [0]*len(maskarray.shape[len(coords):]) 
                    shape = [i + j for i,j in zip(maskarray.shape, coords)]
                    maxshape = tuple([None]*len(shape))
                    self.h5["masks"].create_dataset(
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
                    self.h5["masks"][mask][tuple(slicer)] = maskarray 

        # add tile fields to tiles 
        self.h5["tiles"][str(tile.coords)].attrs["coords"] = str(tile.coords) if tile.coords else 0
        self.h5["tiles"][str(tile.coords)].attrs["name"] = str(tile.name) if tile.coords else 0
        tilelabelsgroup = self.h5["tiles"][tile.coords].create_group("labels")
        for key, val in tile.labels.items():
            self.h5["tiles"][str(tile.coords)]["labels"].attrs[key] = val 
        if tile.counts:
            # cannot concatenate on disk, read into RAM, concatenate, write back to disk
            if self.counts:
                self.counts = self.counts.to_memory()
                self.counts = self.counts.concatenate(tile.counts, join="outer")
                self.counts.filename = os.path.join(self.countspath.name + '/tmpfile.h5ad')
            # cannot concatenate empty AnnData object so set to tile.counts then back in temp file on disk
            else:
                self.counts = tile.counts
                self.counts.filename = os.path.join(self.countspath.name + '/tmpfile.h5ad')

    def update_tile(self, key, val, target):
        """
        Update a tile.

        Args:
            key(str): key of tile to be updated
            val(str): element that will replace target at key
            target(str): element of {all, image, labels} indicating field to be updated 
        """
        key = str(key)
        if key not in self.h5["tiles"].keys():
            raise ValueError(f"key {key} does not exist. Use add.")
        
        if target == 'all':
            assert self.h5["tiles/tile_shape"] == val.image.shape, f"Cannot update a tile of shape {self.h5['tiles/tile_shape']} with a tile" \
                                                           f"of shape {val.image.shape}. Shapes must match."
            # overwrite
            self.add(val)
            print(f'tile at {key} overwritten')

        elif target == 'image':
            assert isinstance(val, np.ndarray), f"when replacing tile image must pass np.ndarray"
            assert self.h5['tiles/tile_shape'] == val.shape, f"Cannot update a tile of shape {self.h5['tiles/tile_shape']} with a tile" \
                                                     f"of shape {val.shape}. Shapes must match."
            coords = list(eval(self.h5['tiles'][key]['coords']))
            slicer = [slice(coords[i], coords[i]+self.h5['tiles/tile_shape'][i]) for i in range(len(coords))] 
            self.h5['array'][tuple(slicer)] = val 
            print(f'array at {key} overwritten')

        elif target == 'masks':
            raise NotImplementedError

        elif target == 'labels':
            assert isinstance(val, (OrderedDict, dict)), f"when replacing labels must pass collections.OrderedDict of labels"
            for key, value in val.items():
                self.h5['tiles'][key]['labels'].attrs['key'] = value 
            print(f'label at {key} overwritten')

        else:
            raise KeyError('target must be all, image, masks, or labels')

    def get_tile(self, item, slicer=None):
        """
        Retrieve tile from h5manager by key or index.

        Args:
            item(int, str, tuple): key or index of tile to be retrieved

        Returns:
            Tile(pathml.core.tile.Tile)
        """
        if isinstance(item, (str, tuple)):
            if str(item) not in self.h5["tiles"].keys():
                raise KeyError(f'key {item} does not exist')
        elif isinstance(item, int):
            if item > len(self.h5["tiles"].keys()) - 1:
                raise KeyError(f'index out of range, valid indices are ints in [0,{len(self.h5["tiles"].keys()) - 1}]')
        else:
            raise KeyError(f'invalid item type: {type(item)}. must getitem by coord (type tuple[int]),'
                           f' index (type int), or name(type str)')
        # impute missing dimensions from self.tile_shape
        coords = list(eval(self.h5["tiles"][item]["coords"]))
        tile_shape = eval(self.h5["tiles/tile_shape"])
        if len(tile_shape) > len(coords):
            shape = list(tile_shape)
            coords = coords + [0]*len(shape[len(coords)-1:]) 
        tiler = [slice(coords[i], coords[i]+tile_shape[i]) for i in range(len(tile_shape))]
        tile = self.h5["array"][tuple(tiler)][:]

        # add masks to tile if there are masks
        if "masks" in self.h5.keys():
            try:
                masks = {mask : self.h5["masks"][mask][tuple(tiler)][:] for mask in self.h5["masks"]}
            except ValueError:
                # if mask is 2-d, need to only use first two dims of tiler
                masks = {mask: self.h5["masks"][mask][tuple(tiler)[0:2]][:] for mask in self.h5["masks"]}
        else:
            masks = None

        if slicer:
            tile = tile[slicer]
            if masks is not None:
                masks = {key : masks[key][slicer] for key in masks}
        masks = pathml.core.masks.Masks(masks)
        labels = {key: val for key, val in self.h5["tiles"][item]["labels"].attrs.items()}
        name = self.h5["tiles"][item]["name"]
        coords = self.h5["tiles"][item]["coords"]
        return pathml.core.tile.Tile(tile, masks=masks, labels=labels, name=name,
                                     coords=coords, slide_type=self.slide_type)

    def slice_tiles(self, slicer):
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
        for key in self.h5["tiles"].keys():
            yield self.get(key, slicer=slicer)
            
    def reshape_tiles(self, shape, centercrop = False):
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
                oldtilecoordlen = len(eval(list(self.h5["tiles"].keys())[0]))
                oldtilecoord = [int(x-y) for x,y in zip(coord, off)]
                oldtilecoord = oldtilecoord[:oldtilecoordlen]
                labels = {key:val for key, val in self.h5["tiles"][str(tuple(oldtilecoord))]["labels"].attrs.items()}
                name = self.h5["tiles"][str(tuple(oldtilecoord))]["name"]
                newtilesdict[str(tuple(coord))] = {
                        'name': None, 
                        'labels': labels,
                        'coords': str(tuple(coord)), 
                        'slidetype': None
                }
            del self.h5["tiles"] 
            self.h5.create_group("tiles")
            for tile in newtilesdict:
                self.h5["tiles"][str(tile)].attrs["coords"] = newtilesdict[str(tile)]["coords"] 
                self.h5["tiles"][str(tile)].attrs["name"] = newtilesdict[str(tile)]["name"] 
                tilelabelsgroup = self.h5["tiles"][str(tile)].create_group("labels")
                for key, val in newtilesdict[str(tile)]["labels"].items():
                    self.h5["tiles"][str(tile.coords)]["labels"].attrs[key] = val 

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
            del self.h5["tiles"] 
            self.h5.create_group("tiles")
            for tile in newtilesdict:
                self.h5["tiles"][str(tile)].attrs["coords"] = newtilesdict[str(tile)]["coords"] 
                self.h5["tiles"][str(tile)].attrs["name"] = newtilesdict[str(tile)]["name"] 
                tilelabelsgroup = self.h5["tiles"][str(tile)].create_group("labels")
                for key, val in newtilesdict[str(tile)]["labels"].items():
                    self.h5["tiles"][str(tile.coords)]["labels"].attrs[key] = val 
        self.tiles = newtilesdict
        self.tile_shape = shape

    def remove_tile(self, key):
        """
        Remove tile from self.h5 by key.
        """
        if not isinstance(key, (str, tuple)):
            raise KeyError(f'key must be str or tuple, check valid keys in repr')
        if str(key) not in self.h5["tiles"].keys():
            raise KeyError(f'key {key} is not in Tiles')
        del self.h5["tiles"][str(key)]

    def add_mask(self, key, mask):
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
        if key in self.h5["masks"].keys():
            raise ValueError(f"key {key} already exists in 'masks'. Cannot add. Must update to modify existing mask.")
        newmask = self.h5["masks"].create_dataset(key, data = mask)

    def update_mask(self, key, mask):
        """
        Update a mask.

        Args:
            key(str): key indicating mask to be updated
            mask(np.ndarray): mask
        """
        if key not in self.h5["masks"].keys():
            raise ValueError(f"key {key} does not exist. Must use add.")
        assert self.h5["masks"][key].shape == mask.shape, f"Cannot update a mask of shape {self.h5['masks'][key].shape}" \
                                                          f" with a mask of shape {mask.shape}. Shapes must match."
        self.h5["masks"][key][...] = mask

    def slice_masks(self, slicer):
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
        for key in self.h5["masks"].keys():
            yield key, self.get_mask(key, slicer=slicer)

    def get_mask(self, item, slicer=None):
        # must check bool separately, since isinstance(True, int) --> True
        if isinstance(item, bool) or not (isinstance(item, str) or isinstance(item, int)):
            raise KeyError(f"key of type {type(item)} must be of type str or int")

        if isinstance(item, str):
            if item not in self.h5["masks"].keys():
                raise KeyError(f'key {item} does not exist')
            if slicer is None:
                return self.h5["masks"][item][:]
            return self.h5["masks"][item][:][tuple(slicer)]

        else:
            try:
                mask_key = list(self.h5.keys())[item]
            except IndexError:
                raise ValueError(f"index out of range, valid indices are ints in [0,{len(self.h5['masks'].keys())}]")
            if slicer is None:
                return self.h5["masks"][mask_key][:]
            return self.h5["masks"][mask_key][:][tuple(slicer)]

    def remove_mask(self, key):
        """
        Remove mask by key.

        Args:
            key(str): key indicating mask to be removed
        """
        if not isinstance(key, str):
            raise KeyError(f"masks keys must be of type(str) but key was passed of type {type(key)}")
        if key not in self.h5["masks"].keys():
            raise KeyError('key is not in Masks')
        del self.h5["masks"][key]

    def get_slidetype(self):
        slide_type_dict = {key:val for key, val in self.h5["fields/slide_type"].items()}
        return pathml.core.slide_types.SlideType(**slide_type_dict)


def check_valid_h5path_format(h5path):
    """
    Assert that the input h5path matches the expected h5path file format:

    Args:
        h5path: h5py file object

    Returns:
        bool: whether the input matches expected format or not
    """
    assert set(h5path.keys()) == {"fields", "array", "masks", "counts", "tiles"}
    assert set(h5path["fields"].keys()) == {"name", "labels", "slide_type"}
    assert set(h5path["fields/slide_type"].keys()) == {"stain", "tma", "rgb", "volumetric", "time_series"}
    return True
