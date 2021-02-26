# h5 utils
import numpy as np

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
