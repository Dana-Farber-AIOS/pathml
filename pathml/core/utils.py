"""
Copyright 2021, Dana-Farber Cancer Institute and Weill Cornell Medicine
License: GNU GPL 2.0
"""

import ast
import tempfile
from collections import OrderedDict
from dataclasses import asdict

import anndata
import h5py
import numpy as np
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
        data=df,
        chunks=True,
        compression="gzip",
        compression_opts=5,
        shuffle=True,
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
        h5[name].attrs.create(str(key), data=val)


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


def writecounts(h5, counts):
    """
    Write counts using anndata h5py.

    Args:
        h5(h5py.Dataset): root of h5 object that counts will be written into
        name(str): name of dataset to be created
        tup(anndata.AnnData): anndata object to be written
    """
    countsh5 = h5py.File(counts.filename, "r")
    for ds in countsh5.keys():
        countsh5.copy(ds, h5)


def readcounts(h5):
    """
    Read counts using anndata h5py.

    Args:
        h5(h5py.Group): h5.Group object from /counts/
    """
    # create and save temp h5py file
    # read using anndata from temp file
    # anndata does not support reading directly from h5
    path = tempfile.NamedTemporaryFile()
    f = h5py.File(path, "w")
    for ds in h5.keys():
        h5.copy(ds, f)
    f.close()
    return anndata.read_h5ad(path.name)
