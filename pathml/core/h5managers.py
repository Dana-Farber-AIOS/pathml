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
        # keep a reference to h5 tempfile so that it is never garbage collected
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
                if ds in ["fields", "array", "masks", "tiles"]:
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
            # fields
            #    create group
            fieldsgroup = self.h5.create_group("fields")
            fieldsgroup.attrs["name"] = slidedata.name
            fieldsgroup.attrs["shape"] = slidedata.slide.get_image_shape()
            labelsgroup = self.h5["fields"].create_group("labels")
            if slidedata.labels:
                for key, label in slidedata.labels.items():
                    self.h5["fields/labels"].attrs[key] = label
            slidetypegroup = self.h5["fields"].create_group("slide_type")
            if slidedata.slide_type:
                for key, val in slidedata.slide_type.asdict().items():
                    self.h5["fields/slide_type"].attrs[key] = val
            # tiles
            tilesgroup = self.h5.create_group("tiles")
            # intitialize tile_shape with zeros
            tilesgroup.attrs["tile_shape"] = b"(0, 0)"
            # array
            self.h5.create_dataset("array", data=h5py.Empty("f"))
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
        Add tile to h5.

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

        # create a group and write tile
        self.h5["tiles"].create_group(str(tile.coords))
        self.h5["tiles"][str(tile.coords)].create_dataset(
            "array",
            data=tile.image,
            # chunks=True,
            # compression="gzip",
            # compression_opts=5,
            # shuffle=True,
            dtype="f16",
        )
        # save tile shape as attribute to enforce consistency
        if "tile_shape" not in self.h5["tiles"].attrs:
            self.h5["tiles"].attrs["tile_shape"] = str(tile.image.shape).encode("utf-8")

        if tile.masks:
            if "masks" not in self.h5["tiles"][str(tile.coords)].keys():
                masksgroup = self.h5["tiles"][str(tile.coords)].create_group("masks")

            for mask in tile.masks:
                # add masks to large mask array
                self.h5["tiles"][str(tile.coords)]["masks"].create_dataset(
                    str(mask), data=tile.masks[mask], dtype="f16",
                )

        # add tile fields to tiles
        self.h5["tiles"][str(tile.coords)].attrs["coords"] = (
            str(tile.coords) if tile.coords else 0
        )
        self.h5["tiles"][str(tile.coords)].attrs["name"] = (
            str(tile.name) if tile.coords else 0
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
            # cannot concatenate empty AnnData object so set to tile.counts then back in temp file on disk
            else:
                self.counts = tile.counts
                self.counts.filename = str(self.countspath.name) + "/tmpfile.h5ad"

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

        if target == "all":
            assert eval(self.h5["tiles"].attrs["tile_shape"]) == val.image.shape, (
                f"Cannot update a tile of shape {self.h5['tiles'].attrs['tile_shape']} with a tile"
                f"of shape {val.image.shape}. Shapes must match."
            )
            # overwrite
            self.remove_tile(key)
            self.add_tile(val)
            print(f"tile at {key} overwritten")

        elif target == "image":
            assert isinstance(
                val, np.ndarray
            ), f"when replacing tile image must pass np.ndarray"
            assert eval(self.h5["tiles"].attrs["tile_shape"]) == val.shape, (
                f"Cannot update a tile of shape {self.h5['tiles'].attrs['tile_shape']} with a tile"
                f"of shape {val.shape}. Shapes must match."
            )
            coords = list(eval(self.h5["tiles"][key].attrs["coords"]))
            slicer = [
                slice(
                    coords[i], coords[i] + eval(self.h5["tiles"].attrs["tile_shape"])[i]
                )
                for i in range(len(coords))
            ]
            self.h5["array"][tuple(slicer)] = val
            print(f"array at {key} overwritten")

        elif target == "masks":
            raise NotImplementedError

        elif target == "labels":
            assert isinstance(
                val, dict
            ), f"when replacing labels must pass collections.OrderedDict of labels"
            for k, v in val.items():
                self.h5["tiles"][key]["labels"].attrs[k] = v
            print(f"label at {key} overwritten")

        else:
            raise KeyError("target must be all, image, masks, or labels")

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

        tile = self.h5["array"][str(item)]

        # add masks to tile if there are masks
        if "masks" in self.h5.keys():
            masks = {
                mask: self.h5["masks"][mask][str(item)] for mask in self.h5["masks"]
            }
        else:
            masks = None

        labels = {
            key: val for key, val in self.h5["tiles"][item]["labels"].attrs.items()
        }
        name = self.h5["tiles"][str(item)].attrs["name"]
        if name == "None":
            name = None
        coords = eval(self.h5["tiles"][str(item)].attrs["coords"])
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

    def add_mask(self, tile, key, mask):
        """
        Add mask to h5.

        Args:
            tile(str or tuple): key indicating tile whose mask will be added
            key(str): key labeling mask
            mask(np.ndarray): mask array
        """
        if not isinstance(mask, np.ndarray):
            raise ValueError(
                f"can not add {type(mask)}, mask must be of type np.ndarray"
            )
        if not isinstance(key, str):
            raise ValueError(f"invalid type {type(key)}, key must be of type str")
        if key in self.h5["tiles"][tile]["masks"].keys():
            raise ValueError(
                f"key {key} already exists in 'masks'. Cannot add. Must update to modify existing mask."
            )
        newmask = self.h5["tiles"][tile]["masks"].create_dataset(key, data=mask)

    def update_mask(self, tile, key, mask):
        """
        Update a mask.

        Args:
            tile(str or tuple): key indicating tile whose mask will be updated
            key(str): key indicating mask to be updated
            mask(np.ndarray): mask
        """
        if key not in self.h5["tiles"][str(tile)]["masks"].keys():
            raise ValueError(f"key {key} does not exist. Must use add.")
        assert self.h5["tiles"][str(tile)]["masks"][key].shape == mask.shape, (
            f"Cannot update a mask of shape {self.h5['masks'][key].shape}"
            f" with a mask of shape {mask.shape}. Shapes must match."
        )
        self.h5["tiles"][tile]["masks"][key][...] = mask

    def get_mask(self, tile, item):
        # must check bool separately, since isinstance(True, int) --> True
        if isinstance(item, bool) or not (
            isinstance(item, str) or isinstance(item, int)
        ):
            raise KeyError(f"key of type {type(item)} must be of type str or int")

        if isinstance(item, str):
            if item not in self.h5["tiles"][tile]["masks"].keys():
                raise KeyError(f"key {item} does not exist")
            return self.h5["tiles"][tile]["masks"][item][:]

        else:
            try:
                mask_key = list(self.h5.keys())[item]
            except IndexError:
                raise ValueError(
                    f"index out of range, valid indices are ints in [0,{len(self.h5['masks'].keys())}]"
                )
            return self.h5["tiles"][tile]["masks"][mask_key][:]

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
    assert set(h5path.keys()) == {"fields", "array", "masks", "counts", "tiles"}
    assert set(h5path["fields"].keys()) == {"labels", "slide_type"}
    assert set(h5path["fields"].attrs.keys()) == {"name", "shape"}
    # slide_type attributes are not enforced
    return True
