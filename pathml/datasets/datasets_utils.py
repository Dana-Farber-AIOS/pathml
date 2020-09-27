import shutil
import urllib.request


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


def download_from_url(url, dest):
    """
    Download a file from a url to destination.
    See: https://stackoverflow.com/questions/7243750/download-file-from-web-in-python-3
    """
    # Download the file from `url` and save it locally under `file_name`:
    with urllib.request.urlopen(url) as response, open(dest, 'wb') as out_file:
        shutil.copyfileobj(response, out_file)

