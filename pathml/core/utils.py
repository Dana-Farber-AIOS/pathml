# h5 utils
import numpy as np


def writedataframeh5(h5, name, df):
    dataset = h5.create_dataset(
        str(name),
        data = df
    )


def writestringh5(h5, name, st):
    stringasarray = np.array(str(st), dtype = object)
    dataset = h5.create_dataset(
        str(name),
        data = stringasarray
    )


def writedicth5(h5, name, dic):
    dictasarray = np.array(list(dic.items()), dtype = object)
    dataset = h5.create_dataset(
        str(name),
        data = dictasarray
    )


def writetupleh5(h5, name, tup):
    tupleasarray = np.array(str(tup), dtype = object)
    dataset = h5.create_dataset(
        str(name),
        data = tupleasarray
    )


# TODO: read h5 methods?
