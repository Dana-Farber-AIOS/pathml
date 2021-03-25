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


def writedicth5(treedict, h5file, h5path='/',
             mode="w", overwrite_data=False,
             create_dataset_args=None):
    """Write a nested dictionary to a HDF5 file, using keys as member names.

    If a dictionary value is a sub-dictionary, a group is created. If it is
    any other data type, it is cast into a numpy array and written as a
    :mod:`h5py` dataset. Dictionary keys must be strings and cannot contain
    the ``/`` character.

        This function requires `h5py <http://www.h5py.org/>`_ to be installed.

    :param treedict: Nested dictionary/tree structure with strings as keys
         and array-like objects as leafs. The ``"/"`` character is not allowed
         in keys.
    :param h5file: HDF5 file name or handle. If a file name is provided, the
        function opens the file in the specified mode and closes it again
        before completing.
    :param h5path: Target path in HDF5 file in which scan groups are created.
        Default is root (``"/"``)
    :param mode: Can be ``"r+"`` (read/write, file must exist),
        ``"w"`` (write, existing file is lost), ``"w-"`` (write, fail if
        exists) or ``"a"`` (read/write if exists, create otherwise).
        This parameter is ignored if ``h5file`` is a file handle.
    :param overwrite_data: If ``True``, existing groups and datasets can be
        overwritten, if ``False`` they are skipped. This parameter is only
        relevant if ``h5file_mode`` is ``"r+"`` or ``"a"``.
    :param create_dataset_args: Dictionary of args you want to pass to
        ``h5f.create_dataset``. This allows you to specify filters and
        compression parameters. Don't specify ``name`` and ``data``.

    Example::

        from silx.io.dictdump import dicttoh5

        city_area = {
            "Europe": {
                "France": {
                    "Isère": {
                        "Grenoble": "18.44 km2"
                    },
                    "Nord": {
                        "Tourcoing": "15.19 km2"
                    },
                },
            },
        }

        create_ds_args = {'compression': "gzip",
                          'shuffle': True,
                          'fletcher32': True}

        dicttoh5(city_area, "cities.h5", h5path="/area",
                 create_dataset_args=create_ds_args)
    """
    if not isinstance(h5file, h5py.File):
        h5f = h5py.File(h5file, mode)
    else:
        h5f = h5file

    if not h5path.endswith("/"):
        h5path += "/"

    for key in treedict:

        if isinstance(treedict[key], dict) and len(treedict[key]):
            # non-empty group: recurse
            dicttoh5(treedict[key], h5f, h5path + key,
                     overwrite_data=overwrite_data,
                     create_dataset_args=create_dataset_args)

        elif treedict[key] is None or (isinstance(treedict[key], dict) and
                                       not len(treedict[key])):
            # Create empty group
            h5f.create_group(h5path + key)

        else:
            ds = _prepare_hdf5_dataset(treedict[key])
            # can't apply filters on scalars (datasets with shape == () )
            if ds.shape == () or create_dataset_args is None:
                h5f.create_dataset(h5path + key,
                                   data=ds)
            else:
                h5f.create_dataset(h5path + key,
                                   data=ds,
                                   **create_dataset_args)

    if isinstance(h5file, string_types):
        h5f.close()

def readdicth5(h5file, path="/", exclude_names=None):
    """Read a HDF5 file and return a nested dictionary with the complete file
    structure and all data.

    If you write a dictionary to a HDF5 file with
    :func:`writedicth5` and then read it back with :func:`readdicth5`, data
    types are not preserved. All values are cast to numpy arrays before
    being written to file, and they are read back as numpy arrays (or
    scalars). In some cases, you may find that a list of heterogeneous
    data types is converted to a numpy array of strings.

    """
    if h5py_missing:
        raise h5py_import_error

    if not is_file(h5file):
        h5f = h5open(h5file)
    else:
        h5f = h5file

    ddict = {}
    for key in h5f[path]:
        if _name_contains_string_in_list(key, exclude_names):
            continue
        if is_group(h5f[path + "/" + key]):
            ddict[key] = h5todict(h5f,
                                  path + "/" + key,
                                  exclude_names=exclude_names)
        else:
            # Convert HDF5 dataset to numpy array
            ddict[key] = h5f[path + "/" + key][...]

    if not is_file(h5file):
        # close file, if we opened it
        h5f.close()

    return ddict

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

