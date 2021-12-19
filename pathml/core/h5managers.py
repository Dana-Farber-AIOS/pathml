"""
Copyright 2021, Dana-Farber Cancer Institute and Weill Cornell Medicine
License: GNU GPL 2.0
"""

import itertools
import os
import tempfile
from collections import OrderedDict

import anndata
import h5py
import numpy as np
import pathml.core
import pathml.core.masks
import pathml.core.tile
from pathml.core.utils import readcounts


class h5pathManager:
    """
    Interface between slidedata object and data management on disk by h5py.
    """

    def __init__(self, h5path=None, slidedata=None):
        path = tempfile.TemporaryFile()
        f = h5py.File(path, "w")
        self.h5 = f
        # keep a reference to the h5 tempfile so that it is never garbage collected
        self.h5reference = path
        # create temporary file for slidedata.counts
        self.countspath = tempfile.TemporaryDirectory()
        self.counts = anndata.AnnData()
        if h5path:
            assert (
                not slidedata
            ), f"if creating h5pathmanager from h5path, slidedata should not be required"
            assert check_valid_h5path_format(
                h5path
            ), f"h5path must conform to .h5path standard, see documentation"
            # copy h5path into self.h5
            for ds in h5path.keys():
                if ds in ["fields", "masks", "tiles"]:
                    h5path.copy(ds, self.h5)
                if ds in ["counts"]:
                    h5path.copy(ds, self.h5)
                    if h5path["counts"].keys():
                        self.counts = readcounts(h5path["counts"])
                        self.counts.filename = (
                            str(self.countspath.name) + "/tmpfile.h5ad"
                        )

        else:
            assert slidedata, f"must pass slidedata object to create h5path"
            # initialize h5path file hierarchy
            # fields
            fieldsgroup = self.h5.create_group("fields")
            # name, shape, labels
            fieldsgroup.attrs["name"] = slidedata.name
            fieldsgroup.attrs["shape"] = slidedata.slide.get_image_shape()
            labelsgroup = self.h5["fields"].create_group("labels")
            if slidedata.labels:
                for key, label in slidedata.labels.items():
                    self.h5["fields/labels"].attrs[key] = label
            # slidetype
            slidetypegroup = self.h5["fields"].create_group("slide_type")
            if slidedata.slide_type:
                for key, val in slidedata.slide_type.asdict().items():
                    self.h5["fields/slide_type"].attrs[key] = val
            # tiles
            tilesgroup = self.h5.create_group("tiles")
            # initialize tile_shape with zeros
            tilesgroup.attrs["tile_shape"] = b"(0, 0)"
            # initialize stride with 0
            tilesgroup.attrs["tile_stride"] = b"(0, 0)"
            # masks
            masksgroup = self.h5.create_group("masks")
            # counts
            countsgroup = self.h5.create_group("counts")

        slide_type_dict = {
            key: val for key, val in self.h5["fields/slide_type"].attrs.items()
        }
        self.slide_type = pathml.core.slide_types.SlideType(**slide_type_dict)

    def __repr__(self):
        rep = f"h5pathManager object, backing a SlideData object named '{self.h5['fields'].attrs['name']}'"
        return rep

    def add_tile(self, tile):
        """
        Add a tile to h5path.

        Args:
            tile(pathml.core.tile.Tile): Tile object
        """
        if str(tile.coords) in self.h5["tiles"].keys():
            print(f"Tile is already in tiles. Overwriting {tile.coords} inplace.")
            # remove old cells from self.counts so they do not duplicate
            if tile.counts:
                if "tile" in self.counts.obs.keys():
                    self.counts = self.counts[self.counts.obs["tile"] != tile.coords]
        # check that the tile matches tile_shape
        existing_shape = eval(self.h5["tiles"].attrs["tile_shape"])
        if all([s == 0 for s in existing_shape]):
            # in this case, tile_shape isn't specified (zeros placeholder)
            # so we set it from the tile image shape
            self.h5["tiles"].attrs["tile_shape"] = str(tile.image.shape).encode("utf-8")
            existing_shape = tile.image.shape

        if any(
            [s1 != s2 for s1, s2 in zip(tile.image.shape[0:2], existing_shape[0:2])]
        ):
            raise ValueError(
                f"cannot add tile of shape {tile.image.shape}. Must match shape of existing tiles: {existing_shape}"
            )

        if self.slide_type and tile.slide_type:
            # check that slide types match
            if tile.slide_type != self.slide_type:
                raise ValueError(
                    f"tile slide_type {tile.slide_type} does not match existing slide_type {self.slide_type}"
                )
        elif not self.slide_type:
            if tile.slide_type:
                self.slide_type = tile.slide_type

        # create a group for tile and write tile
        if str(tile.coords) in self.h5["tiles"]:
            print(f"overwriting tile at {str(tile.coords)}")
            del self.h5["tiles"][str(tile.coords)]
        self.h5["tiles"].create_group(str(tile.coords))
        self.h5["tiles"][str(tile.coords)].create_dataset(
            "array",
            data=tile.image,
            chunks=True,
            compression="gzip",
            compression_opts=5,
            shuffle=True,
            dtype="float16",
        )

        # save tile_shape as an attribute to enforce consistency
        if "tile_shape" not in self.h5["tiles"].attrs or (
            "tile_shape" in self.h5["tiles"].attrs
            and self.h5["tiles"].attrs["tile_shape"] == b"(0, 0)"
        ):
            self.h5["tiles"].attrs["tile_shape"] = str(tile.image.shape).encode("utf-8")

        if tile.masks:
            # create a group to hold tile-level masks
            if "masks" not in self.h5["tiles"][str(tile.coords)].keys():
                masksgroup = self.h5["tiles"][str(tile.coords)].create_group("masks")

            # add tile-level masks
            for key, mask in tile.masks.items():
                self.h5["tiles"][str(tile.coords)]["masks"].create_dataset(
                    str(key),
                    data=mask,
                    dtype="float16",
                )

        # add coords
        self.h5["tiles"][str(tile.coords)].attrs["coords"] = (
            str(tile.coords) if tile.coords else 0
        )
        # add name
        self.h5["tiles"][str(tile.coords)].attrs["name"] = (
            str(tile.name) if tile.name else 0
        )
        tilelabelsgroup = self.h5["tiles"][str(tile.coords)].create_group("labels")
        if tile.labels:
            for key, val in tile.labels.items():
                self.h5["tiles"][str(tile.coords)]["labels"].attrs[key] = val
        if tile.counts:
            # cannot concatenate on disk, read into RAM, concatenate, write back to disk
            if self.counts:
                self.counts = self.counts.to_memory()
                self.counts = self.counts.concatenate(tile.counts, join="outer")
                self.counts.filename = os.path.join(
                    self.countspath.name + "/tmpfile.h5ad"
                )
            # cannot concatenate empty AnnData object so set to tile.counts then set filename
            # so the h5ad object is backed by tempfile
            else:
                self.counts = tile.counts
                self.counts.filename = str(self.countspath.name) + "/tmpfile.h5ad"

    def get_tile(self, item):
        """
        Retrieve tile from h5manager by key or index.

        Args:
            item(int, str, tuple): key or index of tile to be retrieved

        Returns:
            Tile(pathml.core.tile.Tile)
        """
        if isinstance(item, bool):
            raise KeyError(f"invalid key, pass str or tuple")
        if isinstance(item, (str, tuple)):
            item = str(item)
            if item not in self.h5["tiles"].keys():
                raise KeyError(f"key {item} does not exist")
        elif isinstance(item, int):
            if item > len(self.h5["tiles"].keys()) - 1:
                raise IndexError(
                    f'index {item} out of range for total number of tiles: {len(self.h5["tiles"].keys())}'
                )
            item = list(self.h5["tiles"].keys())[item]
        else:
            raise KeyError(
                f"invalid item type: {type(item)}. must getitem by coord (type tuple[int]),"
                f"index (type int), or name (type str)"
            )
        tile = self.h5["tiles"][item]["array"][:]

        # add masks to tile if there are masks
        if "masks" in self.h5["tiles"][item].keys():
            masks = {
                mask: self.h5["tiles"][item]["masks"][mask][:]
                for mask in self.h5["tiles"][item]["masks"]
            }
        else:
            masks = None

        labels = {
            key: val for key, val in self.h5["tiles"][item]["labels"].attrs.items()
        }
        name = self.h5["tiles"][item].attrs["name"]
        if name == "None" or name == 0:
            name = None
        coords = eval(self.h5["tiles"][item].attrs["coords"])

        return pathml.core.tile.Tile(
            tile,
            masks=masks,
            labels=labels,
            name=name,
            coords=coords,
            slide_type=self.slide_type,
        )

    def remove_tile(self, key):
        """
        Remove tile from self.h5 by key.
        """
        if not isinstance(key, (str, tuple)):
            raise KeyError(f"key must be str or tuple, check valid keys in repr")
        if str(key) not in self.h5["tiles"].keys():
            raise KeyError(f"key {key} is not in Tiles")
        del self.h5["tiles"][str(key)]

    def add_mask(self, key, mask):
        """
        Add mask to h5.
        This manages **slide-level masks**.

        Args:
            key(str): mask key
            mask(np.ndarray): mask array
        """
        if not isinstance(mask, np.ndarray):
            raise ValueError(
                f"can not add {type(mask)}, mask must be of type np.ndarray"
            )
        if not isinstance(key, str):
            raise ValueError(f"invalid type {type(key)}, key must be of type str")
        if key in self.h5["masks"].keys():
            raise ValueError(
                f"key {key} already exists in 'masks'. Cannot add. Must update to modify existing mask."
            )
        newmask = self.h5["masks"].create_dataset(key, data=mask)

    def update_mask(self, key, mask):
        """
        Update a mask.

        Args:
            key(str): key indicating mask to be updated
            mask(np.ndarray): mask
        """
        if key not in self.h5["masks"].keys():
            raise ValueError(f"key {key} does not exist. Must use add.")
        assert self.h5["masks"][key].shape == mask.shape, (
            f"Cannot update a mask of shape {self.h5['masks'][key].shape}"
            f" with a mask of shape {mask.shape}. Shapes must match."
        )
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
        if isinstance(item, bool) or not (
            isinstance(item, str) or isinstance(item, int)
        ):
            raise KeyError(f"key of type {type(item)} must be of type str or int")

        if isinstance(item, str):
            if item not in self.h5["masks"].keys():
                raise KeyError(f"key {item} does not exist")
            if slicer is None:
                return self.h5["masks"][item][:]
            return self.h5["masks"][item][:][tuple(slicer)]

        else:
            try:
                mask_key = list(self.h5.keys())[item]
            except IndexError:
                raise ValueError(
                    f"index out of range, valid indices are ints in [0,{len(self.h5['masks'].keys())}]"
                )
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
            raise KeyError(
                f"masks keys must be of type(str) but key was passed of type {type(key)}"
            )
        if key not in self.h5["masks"].keys():
            raise KeyError("key is not in Masks")
        del self.h5["masks"][key]

    def get_slidetype(self):
        slide_type_dict = {
            key: val for key, val in self.h5["fields/slide_type"].items()
        }
        return pathml.core.slide_types.SlideType(**slide_type_dict)


def check_valid_h5path_format(h5path):
    """
    Assert that the input h5path matches the expected h5path file format.

    Args:
        h5path: h5py file object

    Returns:
        bool: True if the input matches expected format
    """
    assert set(h5path.keys()) == {"fields", "masks", "counts", "tiles"}
    assert set(h5path["fields"].keys()) == {"labels", "slide_type"}
    assert set(h5path["fields"].attrs.keys()) == {"name", "shape"}
    assert set(h5path["tiles"].attrs.keys()) == {"tile_shape", "tile_stride"}
    # slide_type attributes are not enforced
    return True
