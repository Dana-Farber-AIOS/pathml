from pathlib import Path, PurePath
import h5py
import os

import pathml.core.slide_data 
from pathml.core.tiles import Tiles
from pathml.core.masks import Masks
from pathml.core.slide_backends import OpenSlideBackend, BioFormatsBackend, DICOMBackend
from pathml.core.utils import writestringh5, writedicth5 

pathmlext = {
    'h5',
    'h5path'
}

openslideext = {
    'svs'
}

bioformatsext = {
    'tiff',
    'tif'
}

dicomext = {
    'dicom'
}


validexts = pathmlext | openslideext | bioformatsext | dicomext

def write_h5path(
    slidedata,
    path
    ):
    """
    Write h5path formatted file from SlideData object.
    
    Args:
        path (str): Path to save destination
    """
    path = Path(path)
    pathdir = Path(os.path.dirname(path)) 
    pathdir.mkdir(parents=True, exist_ok=True) 
    with h5py.File(path, 'w') as f:
        fieldsgroup = f.create_group('fields')
        if slidedata.slide:
            writestringh5(fieldsgroup, 'slide_backend', slidedata.slide_backend)
        if slidedata.name:
            writestringh5(fieldsgroup, 'name', str(slidedata.name))
        if slidedata.labels:
            writedicth5(fieldsgroup, 'labels', slidedata.labels)
        if slidedata.history:
            pass
        if slidedata.masks:
            masksgroup = f.create_group('masks')
            for ds in slidedata.masks.h5manager.h5.keys():
                slidedata.masks.h5manager.h5.copy(ds, f['masks'])
        if slidedata.tiles:
            tilesgroup = f.create_group('tiles')
            for ds in slidedata.tiles.h5manager.h5.keys():
                slidedata.tiles.h5manager.h5.copy(ds, f['tiles'])

def read(
    path,
    backend = None,
    tma = False,
    stain = 'HE'
    ):
    """
    Read file and return :class:`~pathml.slide_data.SlideData` object.

    Args:
        path (str): Path to slide file on disk 
        tma (bool): Flag indicating whether the slide is a tissue microarray 
        stain (str): One of {'HE','IHC','Fluor'}. Flag indicating type of slide stain.
    """
    # TODO: Pass TMA to read. Create slide_class supporting TMAs. 
    # TODO: Pass stain to read. Add stains to class hierarchy. 
    path = Path(path)
    if not path.exists():
        raise ValueError(f"Path does not exist.")
    if is_valid_path(path):
        ext = is_valid_path(path, return_ext = True)
        if backend is None: 
            if ext not in validexts:
                raise ValueError(
                    f"Can only read files with extensions {validexts}. Convert to supported filetype or specify backend."
                )
            if ext in pathmlext: 
                return read_h5path(path)
            elif ext in openslideext:
                return read_openslide(path)
            elif ext in bioformatsext:
                return read_bioformats(path)
            elif ext in dicomext:
                return read_dicom(path)
        elif backend == 'openslide':
            return read_openslide(path)
        elif backend == 'bioformats':
            return read_bioformats(path)
        elif backend == 'dicom':
            return read_dicom(path)
        raise Exception("Must specify valid backend.")

def read_h5path(
    path
    ):
    """
    Read h5path formatted file using h5py and return :class:`~pathml.slide_data.SlideData` object.

    Args:
        path (str): Path to h5path formatted file on disk 
    """
    with h5py.File(path, "r") as f:
        tiles = Tiles(h5 = f['tiles']) if 'tiles' in f.keys() else None 
        masks = Masks(h5 = f['masks']) if 'masks' in f.keys() else None
        backend = f['fields'].attrs['slide_backend'] if 'slide_backend' in f['fields'].attrs.keys() else None
        if backend == "<class 'pathml.core.slide_backend.BioFormatsBackend'>":
            slide_backend = BioformatsBackend
        elif backend == "<class 'pathml.core.slide_backends.DICOMBackend'>":
            slide_backend = DICOMBackend
        else:
            slide_backend = OpenSlideBackend
        name = f['fields'].attrs['name'] if 'name' in f['fields'].attrs.keys() else None
        labels = dict(f['fields'].attrs['labels']) if 'labels' in f['fields'].attrs.keys() else None 
        labels = {k : v for k,v in labels.items()}
        history = None
    return pathml.core.slide_data.SlideData(name = name, slide_backend = slide_backend, masks = masks, tiles = tiles, labels = labels, history = history) 

def read_openslide(
    path
    ):
    """
    Read wsi file using openslide and return :class:`~pathml.slide_data.SlideData` object.

    Args:
        path (str): Path to slide file of supported Openslide format on disk
    """
    return HESlide(filepath = path) 


def read_bioformats(
    path
    ):
    """
    Read bioformat supported imaging format and return :class:`~pathml.slide_data.SlideData` object.

    Args:
        path (str): Path to image file of supported BioFormats format on disk
    """
    return pathml.core.slide_data.SlideData(filepath = path, slide_backend = BioFormatsBackend) 

def read_dicom(
    path
    ):
    """
    Read dicom imaging format and return :class:`~pathml.slide_data.SlideData` object.
    
    Args:
        path (str): Path to image file of supported dicom format on disk
    """
    return pathml.core.slide_data.SlideData(filepath = path, slide_backend = DICOMBackend)

def read_directory(
    tilepath,
    maskpath
    ):
    """
    Read a directory of tiles or masks into a SlideData object. 
    """
    raise NotImplementedError

def is_valid_path(
    path: Path,
    return_ext = False
    ):
    """
    Determine if file format is supported.
    Includes support for compressed files.

    Args:
        path (str): Path to slide file on disk.
        return_ext (bool): If True function return file extension, if False return bool indicating whether the file format is supported. 
    """
    path = Path(path)
    ext = path.suffixes
    if len(ext) > 2:
        print(f"Your path has more than two extensions: {ext}. Only considering {ext[-2]}.")
        ext = ext[-2]

    if len(ext) == 2 and ext[0][1:] in validexts and ext[1][1:] in ('gz', 'bz2'):
        return ext[0][1:] if return_ext else True
    elif ext and ext[-1][1:] in validexts:
        return ext[-1][1:] if return_ext else True
    elif ''.join(ext) == '.soft.gz':
        return 'soft.gz' if return_ext else True
    elif ''.join(ext) == '.mtx.gz':
        return 'mtx.gz' if return_ext else True
    elif not return_ext:
        return False
    raise ValueError(
        f'''\
        {filename!r} does not end on a valid extension.
        Please, provide one of the available extensions.
        {valid}
        Text files with .gz and .bz2 extensions are also supported.\
        '''
    )
