import pytest

from pathml.datasets.utils import parse_file_size, download_from_url


@pytest.mark.parametrize("test_input,expected", [
    ("10 gb", 10**10), ("1.17 mB", 1.17e6), ("0.89 KB", 890)
])
def test_parse_file_sizes(test_input, expected):
    assert parse_file_size(test_input) == expected


def test_download_from_url(tmp_path):
    url = 'http://openslide.cs.cmu.edu/download/openslide-testdata/Aperio/index.yaml'
    d = tmp_path / "test"
    download_from_url(url = url, download_dir = d, name = "testfile")
    file1 = open(d / "testfile", 'r')
    assert file1.readline() == "format: Aperio SVS\n"
