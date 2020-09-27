import bs4
import requests
import os
import re
import pandas as pd

from pathml.datasets.datasets_utils import parse_file_size, download_from_url


def download_openslide_example_data(dest, wsi_format=None, max_file_size="100 Mb"):
    """
    Download all whole-slide images of specified format from
    `openslide test dataset <http://openslide.cs.cmu.edu/download/openslide-testdata/>`_.
    These images are free to use and distribute, and therefore are useful for testing code without worrying about PHI.
    Warning: some of these images may be extremely large (>2GB in some cases). Proceed with caution!

    :param dest: Destination folder for downloaded images.
    :type dest: str
    :param wsi_format: Format of WSI. Must be one of ["Aperio", "Generic-TIFF", "Hamamatsu", "Hamamatsu-vms",
     "Leica", "Mirax", "Olympus", "Trestle", "Ventana", "Zeiss"]. Not case sensitive.
    :type wsi_format: str
    :param max_file_size: Maximum file size to download. Specify as either an integer (number of bytes) or as a string
     containing a number and a standard file size suffix, separated by a space (e.g. \"7.4 Gb\"). Defaults to 100 MB.
    :type max_file_size: str or int, optional
    :return: None
    """
    # open base directory
    url = "http://openslide.cs.cmu.edu/download/openslide-testdata/"
    r = requests.get(url)
    data = bs4.BeautifulSoup(r.text, "html.parser")
    # get available formats
    formats = [re.sub("/$", "", f["href"]) for f in data.find_all("a")]
    for c, item in enumerate(formats):
        if item[-4:] == "json":
            formats.pop(c)

    # make sure input format is a valid option
    wsi_format = wsi_format.lower()
    assert wsi_format in formats, f"supplied wsi_format \'{wsi_format}\' is not valid. Must be one of:\n" \
                                  f" {formats}"

    # navigate to page for specified format, get table of files:
    r2 = requests.get(url + str(wsi_format) + "/")
    data2 = bs4.BeautifulSoup(r2.text, "html.parser").find_all('table')
    df = pd.read_html(str(data2))[0]

    df = df.dropna(subset = ["Size"])

    # convert file sizes into bytes
    df["Size"] = df["Size"].apply(lambda x: parse_file_size(x))

    # get files that are below the size limit
    if isinstance(max_file_size, str):
        size_lim = parse_file_size(max_file_size)
    else:
        size_lim = int(max_file_size)
    df = df.loc[df["Size"] <= size_lim]

    # get list of files to download
    targets = list(df["Name"].values)

    for c, v in enumerate(targets):
        if v[-4:] == "yaml":
            targets.pop(c)
        else:
            targets[c] = os.path.join(url, wsi_format, v)

    # get target directory
    for t in targets:
        download_from_url(t, os.path.join(dest, os.path.basename(t)))
