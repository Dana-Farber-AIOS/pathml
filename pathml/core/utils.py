# h5 utils
import numpy as np
import h5py

# TODO: Fletcher32 checksum?
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
    Write dict as h5 attribute.
    """
    dictasarray = np.array(list(dic.items()), dtype = object)
    h5.attrs[str(name)] = dictasarray


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
        h5.create_group(str(name))

    for key in dic.keys():
        h5.create_group(str(key))
        for key2 in dic[key]:
            stringasarray = np.array(str(dic[key][key2]), dtype = object)
            h5[name].create_dataset(
                str(key2),
                data = stringasarray,
                compression = "gzip",
                compression_opts = 5,
                shuffle = True
            )


def readtilesdicth5(h5):
    """
    Read tilesdict to dict from h5py.Dataset.

    Usage:
        tilesdict = readtilesdicth5(h5['tiles/tilesdict'])
    """
    tilesdict = OrderedDict()
    for tile in h5.keys():
        for field in h5[tile]:
           tilesdict[tile][field] = h5[tile][field].item().decode('UTF-8') 
    return tilesdict
