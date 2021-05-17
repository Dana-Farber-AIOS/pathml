# h5 utils
from collections import OrderedDict
import numpy as np
import h5py
import ast

import pathml.core.slide_backends
import pathml.core.slide_data

# TODO: Fletcher32 checksum?
def writedataframeh5(h5, name, df):
    """
    Write dataframe as h5 dataset.
    Args:
        h5(h5py.Dataset): root of h5 object that df will be written into 
        name(str): name of dataset to be created
        df(pd.DataFrame): dataframe to be written
    """
    dataset = h5.create_dataset(
        str(name),
        data = df,
        chunks = True,
        compression = "gzip",
        compression_opts = 5,
        shuffle = True
    )


def writestringh5(h5, name, st):
    """
    Write string as h5 attribute.
    Args:
        h5(h5py.Dataset): root of h5 object that st will be written into 
        name(str): name of dataset to be created
        st(str): string to be written
    """
    stringasarray = np.string_(str(st))
    h5.attrs[str(name)] = stringasarray


def writedicth5(h5, name, dic):
    """
    Write dict as attributes of h5py.Group. 
    Args:
        h5(h5py.Dataset): root of h5 object that dic will be written into 
        name(str): name of dataset to be created
        dic(str): dict to be written
    """
    h5.create_group(str(name))
    for key, val in dic.items():
        h5[name].attrs.create(
            str(key),
            data = val
        )


def writetupleh5(h5, name, tup):
    """
    Write tuple as h5 attribute.
    Args:
        h5(h5py.Dataset): root of h5 object that tup will be written into 
        name(str): name of dataset to be created
        tup(str): tuple to be written
    """
    tupleasarray = np.string_(str(tup))
    h5.attrs[str(name)] = tupleasarray


def readtupleh5(h5, key):
    """
    Read tuple from h5.
    Args:
        h5(h5py.Dataset or h5py.Group): h5 object that will be read from
        key(str): key where data to read is stored
    """
    return eval(h5.attrs[key]) if key in h5.attrs.keys() else None 


def writetilesdicth5(h5, name, dic):
    """
    Write tiles as h5py.Dataset.
    """
    if name not in h5.keys():
        h5.create_group(str(name), track_order = True)
    assert isinstance(name, (str, tuple)), f"name of h5py.Dataset where tilesdict is written"
    name = str(name)
    for tile in dic.keys():
        tile = str(tile)
        h5[name].create_group(tile, track_order = True)
        for field in dic[tile]:
            # field is name, coords, slidetype
            if isinstance(dic[tile][field], (str, type, type(None))):
                stringasarray = np.string_(str(dic[tile][field]))
                h5[name][tile].create_dataset(
                    field,
                    data = stringasarray,
                    track_order = True
                )
            # field is labels
            elif isinstance(dic[tile][field], (dict, OrderedDict)):
                h5[name][tile].create_group(str(field))
                for key, val in dic[tile][field].items():
                    h5[name][tile][field].attrs.create(
                        str(key),
                        data = val
                    )
            else:
                raise Exception(f"could not write tilesdict element {dic[name][tile]}")


def readtilesdicth5(h5):
    """
    Read tiles into dict from h5py.Dataset.
    Args:
        h5(h5py.Dataset): h5 object that will be read 
    Usage:
        tiles = readtilesdicth5(h5['tiles/tilesdict'])
    """
    tilesdict = OrderedDict()
    for tile in h5.keys():
        name = ast.literal_eval(h5[tile]['name'][...].item().decode('UTF-8')) if 'name' in h5[tile].keys() else None
        labels = h5[tile]['labels'] if 'labels' in h5[tile].keys() else None
        # read the attributes
        if labels:
            labeldict = {}
            # iterate over key/val pairs stored in labels.attr
            for attr in labels.attrs:
                val = labels.attrs[attr]
                # check if val is a single element
                # if val is bytes then decode to str, otherwise leave it (it is a float or int)
                if isinstance(val, bytes):
                    val = val.decode('UTF-8')
                labeldict[attr] = val
            labels = labeldict if labeldict else None
        coords = h5[tile]['coords'][...].item().decode('UTF-8') if 'coords' in h5[tile].keys() else None
        slidetype = h5[tile]['slidetype'][...].item().decode('UTF-8') if 'slidetype' in h5[tile].keys() else None
        # handle slidetype == 'None', must except because strings representing classes will error literal_eval
        # TODO: improve our representation of slidetype (currently just repr)
        if slidetype == 'None':
            slidetype = ast.literal_eval(slidetype)
        if slidetype:
            # TODO: better system for specifying slide classes.
            #  Since it's saved as string here, should have a clean string identifier for each class
            #  currently its using repr essentially
            if slidetype == "<class 'pathml.core.slide_backends.OpenSlideBackend'>":
                slidetype = pathml.core.slide_backends.OpenSlideBackend
            elif slidetype == "<class 'pathml.core.slide_backends.BioFormatsBackend'>":
                slidetype = pathml.core.slide_backends.BioFormatsBackend
            elif slidetype == "<class 'pathml.core.slide_backends.DICOMBackend'>":
                slidetype = pathml.core.slide_backends.DICOMBackend
            elif slidetype == "<class 'pathml.core.slide_classes.HESlide'>":
                slidetype = pathml.core.slide_data.HESlide
        subdict = {
                'name': name,
                'labels': labels,
                'coords': coords,
                'slidetype': slidetype 
        }
        tilesdict[tile] = subdict
    return tilesdict


def writecounts(h5, name, counts):
    """
    Write counts using anndata h5py.
    Args:
        h5(h5py.Dataset): root of h5 object that counts will be written into 
        name(str): name of dataset to be created
        tup(anndata.AnnData): anndata object to be written
    """
    countsh5 = h5py.File(counts.filename, "r") 
    for ds in countsh5.keys():
        countsh5.copy(ds, h5[str(name)])
     

def readcounts(h5):
    """
    Read counts using anndata h5py.
    Args:
        h5(h5py.Dataset): h5 object that will be read 
    """
    # create and save temp h5py file
    # read using anndata from temp file 
    # this is necessary because anndata does not support reading directly from h5
    path = tempfile.TemporaryFile()
    f = h5py.file(path, 'w')
    for ds in h5.keys():
        h5.copy(ds, f)
    return anndata.read_h5ad(path)
