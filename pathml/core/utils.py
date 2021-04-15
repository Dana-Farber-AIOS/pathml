# h5 utils
from collections import OrderedDict
import numpy as np
import h5py
import ast

import pathml.core.slide_classes
import pathml.core.slide_backends


# TODO: Fletcher32 checksum?
import pathml.core.slide_data


def writedataframeh5(h5, name, df):
    """
    Write dataframe as h5 dataset.
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
    """
    stringasarray = np.array(str(st), dtype = object)
    h5.attrs[str(name)] = stringasarray


def writedicth5(h5, name, dic):
    """
    Write dict as h5 dataset. This is not an attribute to accomodate vals that are not strings.
    """
    h5.create_dataset(
        str(name),
        data = str(dic)
    )               


def writetupleh5(h5, name, tup):
    """
    Write tuple as h5 attribute.
    """
    tupleasarray = np.array(str(tup), dtype = object)
    h5.attrs[str(name)] = tupleasarray


def readtupleh5(h5, key):
    """
    Read tuple from h5.
    """
    return eval(h5.attrs[key]) if key in h5.attrs.keys() else None 


def writetilesdicth5(h5, name, dic):
    """
    Write tilesdict as h5py.Dataset.
    """
    if name not in h5.keys():
        h5.create_group(str(name), track_order = True)

    for key in dic.keys():
        h5[str(name)].create_group(str(key))
        for key2 in dic[key]:
            if key2 == 'slidetype':
                stringasarray = np.array(str(dic[key][key2]), dtype = object)
                h5[str(name)][str(key)].create_dataset(
                    str(key2),
                    data = stringasarray
                )
            elif isinstance(dic[key][key2], str):
                stringasarray = np.array(str(dic[key][key2]), dtype = object)
                h5[str(name)][str(key)].create_dataset(
                    str(key2),
                    data = stringasarray
                )
            elif isinstance(dic[key][key2], (dict, OrderedDict)):
                h5[str(name)][str(key)].create_dataset(
                    str(key2),
                    data = str(dic[key][key2])
                )               


def readtilesdicth5(h5):
    """
    Read tilesdict to dict from h5py.Dataset.

    Usage:
        tilesdict = readtilesdicth5(h5['tiles/tilesdict'])
    """
    tilesdict = OrderedDict()
    for tile in h5.keys():
        name = h5[tile]['name'][...].item().decode('UTF-8') if 'name' in h5[tile].keys() else None
        # labels = dict(h5[tile]['labels']) if 'labels' in h5[tile].keys() else None 
        labels = h5[tile].get('labels')[...].tolist() if 'labels' in h5[tile].keys() else None 
        coords = h5[tile]['coords'][...].item().decode('UTF-8') if 'coords' in h5[tile].keys() else None
        slidetype = h5[tile]['slidetype'][...].item().decode('UTF-8') if 'slidetype' in h5[tile].keys() else None
        if slidetype:
            if slidetype == "<class 'pathml.core.slide_backends.OpenSlideBackend'>":
                slidetype = pathml.core.slide_backends.OpenSlideBackend
            elif slidetype == "<class 'pathml.core.slide_backends.BioFormatsBackend'>":
                slidetype = pathml.core.slide_backends.BioFormatsBackend
            elif slidetype == "<class 'pathml.core.slide_backends.DICOMBackend'>":
                slidetype = pathml.core.slide_backends.DICOMBackend
            elif slidetype == "<class 'pathml.core.slide_classes.HESlide'>":
                slidetype = pathml.core.slide_data.HESlide
        if labels:
            labels = ast.literal_eval(labels.decode('UTF-8')) 
            print(f"labels are {labels}")
        subdict = {
                'name': name,
                'labels': labels,
                'coords': coords,
                'slidetype': slidetype 
        }
        tilesdict[tile] = subdict
    return tilesdict
