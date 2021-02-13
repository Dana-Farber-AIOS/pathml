from pathlib import Path, PurePath

from pathml.core.slide_data import SlideData

pathmlext = {
    'h5',
    'h5path'
}

openslideext = {
    'svs'
}

bioformatsext = {
    'tiff'
}


validexts = pathmlext | openslideext | bioformatsext

def write_h5path(
    slidedata,
    path
    ) -> None:
    f = h5py.File(path, 'w')
    fieldsgroup = f.create_group('fields')
    masksgroup = f.create_group('masks')
    tilesgroup = f.create_Group('tiles')
    if slidedata.slide:
        writestringh5(fieldsgroup, 'slide_backend', str(type(slidedata.slide)))
    if slidedata.name:
        writestringh5(fieldsgroup, 'name', str(slidedata.name))
    if slidedata.labels:
        writedicth5(fieldsgroup, 'labels', slidedata.labels)
    if slidedata.history:
        pass
    if slidedata.masks:
        for ds in slidedata.masks.keys():
            # slidedata.masks.copy(ds, f['masks'])
            slidedata.masks.copy(ds, f)
    if slidedata.tiles:
        for ds in slidedata.tiles.keys():
            # slidedata.tiles.copy(ds, f['tiles'])
            slidedata.tiles.copy(ds, f)

def read(
    path
    ) -> SlideData:
    """
    Read file and return :class:`~pathml.slide_data.SlideData` object.

    Args:
        path (str): Path to slide file on disk 
    """
    path = Path(path)
    if not path.exists():
        raise ValueError(f"Path does not exist.")
    if is_valid_path(path):
        ext = is_valid_path(path)
            if ext not in validexts:
                raise ValueError(
                    f"Can only read files with extensions {validexts}"
                )
            if ext in pathmlext: 
                return read_h5path(path)
            elif ext in openslideext:
                return read_openslide(path)
            elif exit in bioformatsext:
                return read_bioformats(path)

def read_h5path(
    path
    ) -> SlideData:
    """
    Read h5path formatted file using h5py and return :class:`~pathml.slide_data.SlideData` object.
    """
    with h5py.File(path, "r") as f:
        tiles = f['tiles'] if 'tiles' in f.keys() else None 
        masks = f['masks'] if 'masks' in f.keys() else None
        slide_backend = str(f['fields/slide_backend'][...]) if 'slide_backend' in f['fields'].keys() else None
        name = str(f['fields/name'][...]) if 'name' in f['fields'].keys() else None
        labels = dict(f['fields/labels'][...]) if 'labels' in f['fields'].keys() else None 
        # TODO: implement history
        history = None
    return SlideData(name = name, slide_backend = slide_backend, masks = masks, tiles = tiles, labels = labels) 

def read_openslide(
    path
    ) -> SlideData:
    """
    Read wsi file using openslide and return :class:`~pathml.slide_data.SlideData` object.
    """
    return HESlide(filepath = path) 


def read_bioformats(
    path
    ) -> SlideData:
    """
    Read bioformat supported imaging format and return :class:`~pathml.slide_data.SlideData` object.
    """
    return SlideData(filepath = path, slide_backend='Bioformats') 

def read_directory(
    tilepath,
    maskpath
    ) -> SlideData:
    """
    Read slidedata files from directories of tile and mask objects. 
    """
    raise NotImplementedError

def is_valid_path(
    path: Path
    return_ext = False
    ):
    """
    Determine if file format is supported.
    Includes support for compressed files.
    Args:
        path (str): Path to slide file on disk.
        return_ext (bool): If True function return file extension, if False return bool indicating whether the file format is supported. 
    """
    ext = filename.suffixes
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
