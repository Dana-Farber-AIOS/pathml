"""
Copyright 2021, Dana-Farber Cancer Institute and Weill Cornell Medicine
License: GNU GPL 2.0
"""

import os
import reprlib
from pathlib import Path

import anndata
import dask.distributed
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pathml.core
import pathml.preprocessing.pipeline
from pathml.core.slide_types import SlideType
from torch.utils.data import Dataset


def get_file_ext(path):
    """
    Return the file extension of an input path.
    If zipped with 'gz' or 'bz2' extension, will instead return the second to last extension.
    If multiple extensions, will return the last two.

    Args:
        path: path to file

    Returns:
        str: file extension
    """
    p = Path(path)
    ext = p.suffixes
    if not ext:
        raise Exception(f"invalid path has no file extension: {path}")
    elif len(ext) == 1:
        ext = ext[0]
    elif len(ext) >= 2:
        if ext[-1] in {".gz", ".bz2"}:
            ext = ext[-2]
        else:
            ext = "".join(ext[-2:])
    return ext


class SlideData:
    """
    Main class representing a slide and its annotations.

    Args:
        filepath (str): Path to file on disk.
        name (str, optional): name of slide. If ``None``, and a ``filepath`` is provided, name defaults to filepath.
        masks (pathml.core.Masks, optional): object containing {key, mask} pairs
        tiles (pathml.core.Tiles, optional): object containing {coordinates, tile} pairs
        labels (collections.OrderedDict, optional): dictionary containing {key, label} pairs
        backend (str, optional): backend to use for interfacing with slide on disk.
            Must be one of {"OpenSlide", "BioFormats", "DICOM"} (case-insensitive).
            If ``None``, and a ``filepath`` is provided, tries to infer the correct backend from the file extension.
        slide_type (pathml.core.SlideType, optional): slide type specification. Must be a
            :class:`~pathml.core.SlideType` object. Alternatively, slide type can be specified by using the
            parameters ``stain``, ``tma``, ``rgb``, ``volumetric``, and ``time_series``.
        stain (str, optional): Flag indicating type of slide stain. Must be one of [‘HE’, ‘IHC’, ‘Fluor’].
            Defaults to ``None``. Ignored if ``slide_type`` is specified.
        platform (str, optional): Flag indicating the imaging platform (e.g. CODEX, Vectra, etc.).
            Defaults to ``None``. Ignored if ``slide_type`` is specified.
        tma (bool, optional): Flag indicating whether the image is a tissue microarray (TMA).
            Defaults to ``False``. Ignored if ``slide_type`` is specified.
        rgb (bool, optional): Flag indicating whether the image is in RGB color.
            Defaults to ``None``. Ignored if ``slide_type`` is specified.
        volumetric (bool, optional): Flag indicating whether the image is volumetric.
            Defaults to ``None``. Ignored if ``slide_type`` is specified.
        time_series (bool, optional): Flag indicating whether the image is a time series.
            Defaults to ``None``. Ignored if ``slide_type`` is specified.
        counts (anndata.AnnData): object containing counts matrix associated with image quantification
    """

    def __init__(
        self,
        filepath,
        name=None,
        masks=None,
        tiles=None,
        labels=None,
        backend=None,
        slide_type=None,
        stain=None,
        platform=None,
        tma=None,
        rgb=None,
        volumetric=None,
        time_series=None,
        counts=None,
    ):
        # check inputs
        assert masks is None or isinstance(
            masks, dict
        ), f"mask are of type {type(masks)} but must be type dict"
        if labels:
            assert all(
                [isinstance(key, str) for key in labels.keys()]
            ), f"Input label keys are of types {[type(k) for k in labels.keys()]}. All label keys must be of type str."
            assert all(
                [
                    isinstance(val, (str, np.ndarray))
                    or np.issubdtype(type(val), np.number)
                    or np.issubdtype(type(val), np.bool_)
                    for val in labels.values()
                ]
            ), (
                f"Input label vals are of types {[type(v) for v in labels.values()]}. "
                f"All label values must be of type str or np.ndarray or a number (i.e. a subdtype of np.number) "
            )
        assert tiles is None or (
            isinstance(tiles, list)
            and all([isinstance(tile, pathml.core.Tile) for tile in tiles])
        ), f"tiles are of type {type(tiles)} but must be a list of objects of type pathml.core.tiles.Tile"
        assert slide_type is None or isinstance(
            slide_type, pathml.core.SlideType
        ), f"slide_type is of type {type(slide_type)} but must be of type pathml.core.types.SlideType"
        assert backend is None or (
            isinstance(backend, str)
            and backend.lower() in {"openslide", "bioformats", "dicom"}
        ), f"backend {backend} must be one of ['OpenSlide', 'BioFormats', 'DICOM'] (case-insensitive)."
        assert counts is None or isinstance(
            counts, anndata.AnnData
        ), f"counts is if type {type(counts)} but must be of type anndata.AnnData"

        # instantiate SlideType object if needed
        if not slide_type and any([stain, platform, tma, rgb, volumetric, time_series]):
            stain_type_dict = {
                "stain": stain,
                "platform": platform,
                "tma": tma,
                "rgb": rgb,
                "volumetric": volumetric,
                "time_series": time_series,
            }
            # remove any Nones
            stain_type_dict = {
                key: val for key, val in stain_type_dict.items() if val is not None
            }
            if stain_type_dict:
                slide_type = pathml.core.slide_types.SlideType(**stain_type_dict)

        # get name from filepath if no name is provided
        if name is None and filepath is not None:
            name = Path(filepath).stem

        _load_from_h5path = False

        if backend:
            # convert everything to lower so it's case insensitive
            backend = backend.lower()
        else:
            # try to infer the correct backend
            ext = get_file_ext(filepath)
            if ext in openslideext:
                backend = "openslide"
            elif ext in bioformatsext:
                backend = "bioformats"
            elif ext in dicomext:
                backend = "dicom"
            elif ext in pathmlext:
                backend = "h5path"
                # load SlideData from h5 or h5path
                _load_from_h5path = True
            else:
                raise ValueError(
                    f"Backend not specified, but cannot infer correct backend from input path {filepath}"
                )

        if backend.lower() == "openslide":
            backend_obj = pathml.core.OpenSlideBackend(filepath)
        elif backend.lower() == "bioformats":
            backend_obj = pathml.core.BioFormatsBackend(filepath)
        elif backend.lower() == "dicom":
            backend_obj = pathml.core.DICOMBackend(filepath)
        elif backend.lower() == "h5path":
            backend_obj = None
        else:
            raise ValueError(f"invalid backend: {repr(backend)}.")

        self._filepath = filepath if filepath else None
        self.backend = backend
        self.slide = backend_obj if backend_obj else None
        self.name = name
        self.labels = labels
        self.slide_type = slide_type

        if _load_from_h5path:
            # populate the SlideData object from existing h5path file
            with h5py.File(filepath, "r") as f:
                self.h5manager = pathml.core.h5managers.h5pathManager(h5path=f)
            self.name = self.h5manager.h5["fields"].attrs["name"]
            self.labels = {
                key: val
                for key, val in self.h5manager.h5["fields"]["labels"].attrs.items()
            }
            # empty dict evaluates to False
            if not self.labels:
                self.labels = None
            slide_type = {
                key: val
                for key, val in self.h5manager.h5["fields"]["slide_type"].attrs.items()
                if val is not None
            }
            if slide_type:
                self.slide_type = SlideType(**slide_type)
        else:
            self.h5manager = pathml.core.h5managers.h5pathManager(slidedata=self)

        self.masks = pathml.core.Masks(h5manager=self.h5manager, masks=masks)
        self.tiles = pathml.core.Tiles(h5manager=self.h5manager, tiles=tiles)

    def __repr__(self):
        out = []
        out.append(f"SlideData(name={repr(self.name)}")
        out.append(f"slide_type={repr(self.slide_type)}")
        if self._filepath:
            out.append(f"filepath='{self._filepath}'")
        if self.backend:
            out.append(f"backend={repr(self.backend)}")
        out.append(f"image shape: {self.shape}")
        out.append(repr(self.tiles))
        out.append(repr(self.masks))
        if self.tiles:
            out.append(f"tile_shape={eval(self.tiles.tile_shape)}")
        if self.labels:
            out.append(
                f"{len(self.labels)} labels: {reprlib.repr(list(self.labels.keys()))}"
            )
        else:
            out.append("labels=None")
        if self.counts:
            out.append(f"counts matrix of shape {self.counts.shape}")
        else:
            out.append(f"counts=None")

        out = ",\n\t".join(out)
        out += ")"
        return out

    def run(
        self,
        pipeline,
        distributed=True,
        client=None,
        tile_size=3000,
        tile_stride=None,
        level=0,
        tile_pad=False,
        overwrite_existing_tiles=False,
    ):
        """
        Run a preprocessing pipeline on SlideData.
        Tiles are generated by calling self.generate_tiles() and pipeline is applied to each tile.

        Args:
            pipeline (pathml.preprocessing.pipeline.Pipeline): Preprocessing pipeline.
            distributed (bool): Whether to distribute model using client. Defaults to True.
            client: dask.distributed client
            tile_size (int, optional): Size of each tile. Defaults to 3000px
            tile_stride (int, optional): Stride between tiles. If ``None``, uses ``tile_stride = tile_size``
                for non-overlapping tiles. Defaults to ``None``.
            level (int, optional): Level to extract tiles from. Defaults to ``None``.
            tile_pad (bool): How to handle chunks on the edges. If ``True``, these edge chunks will be zero-padded
                symmetrically and yielded with the other chunks. If ``False``, incomplete edge chunks will be ignored.
                Defaults to ``False``.
            overwrite_existing_tiles (bool): Whether to overwrite existing tiles. If ``False``, running a pipeline will
                fail if ``tiles is not None``. Defaults to ``False``.
        """
        assert isinstance(
            pipeline, pathml.preprocessing.pipeline.Pipeline
        ), f"pipeline is of type {type(pipeline)} but must be of type pathml.preprocessing.pipeline.Pipeline"
        assert self.slide is not None, "cannot run pipeline because self.slide is None"

        if len(self.tiles) != 0:
            # in this case, tiles already exist
            if not overwrite_existing_tiles:
                raise Exception(
                    "Slide already has tiles. Running the pipeline will overwrite the existing tiles."
                    "use overwrite_existing_tiles=True to force overwriting existing tiles."
                )
            else:
                # delete all existing tiles
                for tile_key in self.tiles.keys:
                    self.tiles.remove(tile_key)

        # TODO: be careful here since we are modifying h5 outside of h5manager
        # look into whether we can push this into h5manager
        self.h5manager.h5["tiles"].attrs["tile_stride"] = tile_stride

        if distributed:
            if client is None:
                client = dask.distributed.Client()

            # map pipeline application onto each tile
            processed_tile_futures = []

            for tile in self.generate_tiles(
                level=level, shape=tile_size, stride=tile_stride, pad=tile_pad
            ):
                if not tile.slide_type:
                    tile.slide_type = self.slide_type
                # explicitly scatter data, i.e. send the tile data out to the cluster before applying the pipeline
                # according to dask, this can reduce scheduler burden and keep data on workers
                big_future = client.scatter(tile)
                f = client.submit(pipeline.apply, big_future)
                processed_tile_futures.append(f)

            # as tiles are processed, add them to h5
            for future, tile in dask.distributed.as_completed(
                processed_tile_futures, with_results=True
            ):
                self.tiles.add(tile)

        else:
            for tile in self.generate_tiles(
                level=level, shape=tile_size, stride=tile_stride, pad=tile_pad
            ):
                if not tile.slide_type:
                    tile.slide_type = self.slide_type
                pipeline.apply(tile)
                self.tiles.add(tile)

    @property
    def shape(self):
        """
        Convenience method for getting the image shape.
        Calling ``wsi.shape`` is equivalent to calling ``wsi.slide.get_image_shape()`` with default arguments.

        Returns:
            Tuple[int, int]: Shape of image (H, W)
        """
        if self.backend == "h5path":
            return tuple(self.h5manager.h5["fields"].attrs["shape"])
        else:
            return self.slide.get_image_shape()

    def generate_tiles(self, shape=3000, stride=None, pad=False, **kwargs):
        """
        Generator over Tile objects containing regions of the image.
        Calls ``generate_tiles()`` method of the backend.
        Tries to add the corresponding slide-level masks to each tile, if possible.
        Adds slide-level labels to each tile, if possible.

        Args:
            shape (int or tuple(int)): Size of each tile. May be a tuple of (height, width) or a single integer,
                in which case square tiles of that size are generated.
            stride (int): stride between chunks. If ``None``, uses ``stride = size`` for non-overlapping chunks.
                Defaults to ``None``.
            pad (bool): How to handle tiles on the edges. If ``True``, these edge tiles will be zero-padded
                and yielded with the other chunks. If ``False``, incomplete edge chunks will be ignored.
                Defaults to ``False``.
            **kwargs: Other arguments passed through to ``generate_tiles()`` method of the backend.

        Yields:
            pathml.core.tile.Tile: Extracted Tile object
        """
        for tile in self.slide.generate_tiles(shape, stride, pad, **kwargs):
            # add masks for tile, if possible
            # i.e. if the SlideData has a Masks object, and the tile has coordinates
            if self.masks is not None and tile.coords is not None:
                # masks not supported if pad=True
                # to implement, need to update Mask.slice to support slices that go beyond the full mask
                if not pad:
                    i, j = tile.coords
                    di, dj = tile.image.shape[0:2]
                    # add the Masks object for the masks corresponding to the tile
                    # this assumes that the tile didn't already have any masks
                    # this should work since the backend reads from image only
                    # adding safety check just in case to make sure we don't overwrite any existing mask
                    # if this assertion fails, we will need to rewrite this part
                    assert (
                        len(tile.masks) == 0
                    ), "tile yielded from backend already has mask. slide_data.generate_tiles is trying to overwrite it"

                    tile_slices = [slice(i, i + di), slice(j, j + dj)]
                    tile.masks = self.masks.slice(tile_slices)

            # add slide-level labels to each tile, if possible
            if self.labels is not None:
                tile.labels = self.labels

            # add slidetype to tile
            if tile.slide_type is None:
                tile.slide_type = self.slide_type

            yield tile

    def plot(self, ax=None):
        """
        View a thumbnail of the image, using matplotlib.
        Not supported by all backends.

        Args:
            ax: matplotlib axis object on which to plot the thumbnail. Optional.
        """
        try:
            thumbnail = self.slide.get_thumbnail(size=(500, 500))
        except:
            if not self.slide:
                raise NotImplementedError(
                    "Plotting only supported via backend, but SlideData has no backend."
                )
            else:
                raise NotImplementedError(
                    f"plotting not supported for slide_backend={self.slide.__class__.__name__}"
                )
        if ax is None:
            ax = plt.gca()
        ax.imshow(thumbnail)
        if self.name:
            ax.set_title(self.name)
        ax.axis("off")

    @property
    def counts(self):
        return self.tiles.h5manager.counts if self.tiles.h5manager else None

    @counts.setter
    def counts(self, value):
        if self.tiles.h5manager:
            assert value is None or isinstance(
                value, anndata.AnnData
            ), f"cannot set counts with obj of type {type(value)}. Must be Anndata"
            self.tiles.h5manager.counts = value
        else:
            raise AttributeError(
                "cannot assign counts slidedata contains no tiles, first generate tiles"
            )

    def write(self, path):
        """
        Write contents to disk in h5path format.

        Args:
            path (Union[str, bytes, os.PathLike]): path to file to be written
        """
        path = Path(path)
        pathdir = Path(os.path.dirname(path))
        pathdir.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w") as f:
            for ds in self.h5manager.h5.keys():
                self.h5manager.h5.copy(ds, f)
            if self.counts:
                pathml.core.utils.writecounts(f["counts"], self.counts)


class HESlide(SlideData):
    """
    Convenience class to load a SlideData object for H&E slides.
    Passes through all arguments to ``SlideData()``, along with ``slide_type = types.HE`` flag.
    Refer to :class:`~pathml.core.slide_data.SlideData` for full documentation.
    """

    def __init__(self, *args, **kwargs):
        kwargs["slide_type"] = pathml.core.types.HE
        super().__init__(*args, **kwargs)


class MultiparametricSlide(SlideData):
    """
    Convenience class to load a SlideData object for multiparametric immunofluorescence slides.
    Passes through all arguments to ``SlideData()``, along with ``slide_type = types.IF`` flag and default ``backend = "bioformats"``.
    Refer to :class:`~pathml.core.slide_data.SlideData` for full documentation.
    """

    def __init__(self, *args, **kwargs):
        kwargs["slide_type"] = pathml.core.types.IF
        if "backend" not in kwargs:
            kwargs["backend"] = "bioformats"
        super().__init__(*args, **kwargs)


class IHCSlide(SlideData):
    """
    Convenience class to load a SlideData object for IHC slides.
    Passes through all arguments to ``SlideData()``, along with ``slide_type = types.IHC`` flag.
    Refer to :class:`~pathml.core.slide_data.SlideData` for full documentation.
    """

    def __init__(self, *args, **kwargs):
        kwargs["slide_type"] = pathml.core.types.IHC
        super().__init__(*args, **kwargs)


class VectraSlide(SlideData):
    """
    Convenience class to load a SlideData object for Vectra (Polaris) slides.
    Passes through all arguments to ``SlideData()``, along with ``slide_type = types.Vectra`` flag and default ``backend = "bioformats"``.
    Refer to :class:`~pathml.core.slide_data.SlideData` for full documentation.
    """

    def __init__(self, *args, **kwargs):
        kwargs["slide_type"] = pathml.core.types.Vectra
        if "backend" not in kwargs:
            kwargs["backend"] = "bioformats"
        super().__init__(*args, **kwargs)


class CODEXSlide(SlideData):
    """
    Convenience class to load a SlideData object from Akoya Biosciences CODEX format.
    Passes through all arguments to ``SlideData()``, along with ``slide_type = types.CODEX`` flag and default ``backend = "bioformats"``.
    Refer to :class:`~pathml.core.slide_data.SlideData` for full documentation.

    # TODO:
        hierarchical biaxial gating (flow-style analysis)
    """

    def __init__(self, *args, **kwargs):
        kwargs["slide_type"] = pathml.core.types.CODEX
        if "backend" not in kwargs:
            kwargs["backend"] = "bioformats"
        super().__init__(*args, **kwargs)


# dicts used to infer correct backend from file extension

pathmlext = {".h5", ".h5path"}

openslideext = {
    ".svs",
    ".tif",
    ".ndpi",
    ".vms",
    ".vmu",
    ".scn",
    ".mrxs",
    ".svslide",
    ".bif",
}

bioformatsext = {
    ".tiff",
    ".tif",
    ".sld",
    ".aim",
    ".al3d",
    ".gel",
    ".am",
    ".amiramesh",
    ".grey",
    ".hx",
    ".labels",
    ".cif",
    ".img",
    ".hdr",
    ".sif",
    ".png",
    ".afi",
    ".htd",
    ".pnl",
    ".avi",
    ".arf",
    ".exp",
    ".spc",
    ".sdt",
    ".xml",
    ".1sc",
    ".pic",
    ".raw",
    ".scn",
    ".ims",
    ".img",
    ".cr2",
    ".crw",
    ".ch5",
    ".c01",
    ".dib",
    ".vsi",
    ".wpi",
    ".dv",
    ".r3d",
    ".rcpnl",
    ".eps",
    ".epsi",
    ".ps",
    ".fits",
    ".dm3",
    ".dm4",
    ".dm2",
    ".vff",
    ".naf",
    ".his",
    ".i2i",
    ".ics",
    ".ids",
    ".fff",
    ".seq",
    ".ipw",
    ".hed",
    ".mod",
    ".liff",
    ".obf",
    ".msr",
    ".xdce",
    ".frm",
    ".inr",
    ".hdr",
    ".ipl",
    ".ipm",
    ".dat",
    ".par",
    ".jp2",
    ".j2k",
    ".jpf",
    ".jpk",
    ".jpx",
    ".klb",
    ".xv",
    ".bip",
    ".fli",
    ".msr",
    ".lei",
    ".lif",
    ".scn",
    ".sxm",
    ".l2d",
    ".lim",
    ".stk",
    ".nd",
    ".htd",
    ".mnc",
    ".mrw",
    ".mng",
    ".stp",
    ".mrc",
    ".st",
    ".ali",
    ".map",
    ".rec",
    ".mrcs",
    ".nef",
    ".hdr",
    ".nii",
    ".nii.gz",
    ".nrrd",
    ".nhdr",
    ".apl",
    ".mtb",
    ".tnb",
    ".obsep",
    ".oib",
    ".oif",
    ".oir",
    ".ome.tiff",
    ".ome.tif",
    ".ome.tf2",
    ".ome.tf8",
    ".ome.btf",
    ".ome.xml",
    ".ome",
    ".top",
    ".pcoraw",
    ".rec",
    ".pcx",
    ".pds",
    ".im3",
    ".qptiff",
    ".pbm",
    ".pgm",
    ".ppm",
    ".psd",
    ".bin",
    ".pict",
    ".cfg",
    ".spe",
    ".afm",
    ".mov",
    ".sm2",
    ".sm3",
    ".xqd",
    ".xqf",
    ".cxd",
    ".spi",
    ".stk",
    ".tga",
    ".db",
    ".vws",
    ".tfr",
    ".ffr",
    ".zfr",
    ".zfp",
    ".2fl",
    ".sld",
    ".pr3",
    ".dat",
    ".hdr",
    ".fdf",
    ".bif",
    ".dti",
    ".xys",
    ".mvd2",
    ".acff",
    ".wat",
    ".bmp",
    ".wlz",
    ".lms",
    ".zvi",
    ".czi",
    ".lsm",
    ".mdb",
}

dicomext = {".dicom", ".dcm"}
