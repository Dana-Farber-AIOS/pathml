import os
import shutil
import urllib


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