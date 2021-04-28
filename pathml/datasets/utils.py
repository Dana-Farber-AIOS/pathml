"""
Copyright 2021, Dana-Farber Cancer Institute and Weill Cornell Medicine
License: GNU GPL 2.0
"""

import os
import shutil
import urllib
import numpy as np


def parse_file_size(fs):
    """
    Parse a file size string into bytes.
    """
    units = {"B": 1,
             "KB": 10 ** 3,
             "MB": 10 ** 6,
             "GB": 10 ** 9,
             "TB": 10 ** 12}
    number, unit = [s.strip() for s in fs.split()]
    return int(float(number) * units[unit.upper()])


def download_from_url(url, download_dir, name=None):
    """
    Download a file from a url to destination directory.
    If the file already exists, does not download.

    Args:
        url (str): Url of file to download
        download_dir (str): Directory where file will be downloaded
        name (str, optional): Name of saved file. If ``None``, uses base name of url argument. Defaults to ``None``.

    See: https://stackoverflow.com/questions/7243750/download-file-from-web-in-python-3
    """
    if name is None:
        name = os.path.basename(url)

    path = os.path.join(download_dir, name)

    if os.path.exists(path):
        return
    else:
        os.makedirs(download_dir, exist_ok = True)

        # Download the file from `url` and save it locally under `file_name`:
        with urllib.request.urlopen(url) as response, open(path, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)


def pannuke_multiclass_mask_to_nucleus_mask(multiclass_mask):
    """
    Convert multiclass mask from PanNuke to a single channel nucleus mask.
    Assumes each pixel is assigned to one and only one class. Sums across channels, except the last mask channel
    which indicates background pixels in PanNuke.
    Operates on a single mask.

    Args:
        multiclass_mask (torch.Tensor): Mask from PanNuke, in classification setting. (i.e. ``nucleus_type_labels=True``).
            Tensor of shape (6, 256, 256).

    Returns:
        Tensor of shape (256, 256).
    """
    # verify shape of input
    assert multiclass_mask.ndim == 3 and multiclass_mask.shape[0] == 6, \
        f"Expecting a mask with dims (6, 256, 256). Got input of shape {multiclass_mask.shape}"
    assert multiclass_mask.shape[1] == 256 and multiclass_mask.shape[2] == 256, \
        f"Expecting a mask with dims (6, 256, 256). Got input of shape {multiclass_mask.shape}"
    # ignore last channel
    out = np.sum(multiclass_mask[:-1, :, :], axis = 0)
    return out