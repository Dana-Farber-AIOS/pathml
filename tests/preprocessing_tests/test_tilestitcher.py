import glob
import os
import subprocess
import tempfile
import zipfile
from unittest.mock import MagicMock, patch

import javabridge
import jpype
import pytest

from pathml.preprocessing.tilestitcher import TileStitcher
from pathml.utils import setup_qupath


@pytest.mark.exclude
@pytest.fixture(scope="module")
def tile_stitcher(request):
    try:
        javabridge.kill_vm()
        print("Javabridge vm terminated")
    except Exception as e:
        print(f"JVM isn't running: {e}")
        pass  # JVM was not running, so nothing to kill

    # Set JAVA_HOME
    # os.environ["JAVA_HOME"] = "/usr/lib/jvm/jdk-17/"

    # Setup QuPath using the setup_qupath function
    qupath_home = setup_qupath(
        "../../tools/qupath"
    )  # Replace with the appropriate path

    # Ensure QUPATH_HOME is set
    os.environ["QUPATH_HOME"] = qupath_home

    # Construct path to QuPath jars
    qupath_jars_dir = os.path.join(qupath_home, "lib", "app")
    qupath_jars = glob.glob(os.path.join(qupath_jars_dir, "*.jar"))
    qupath_jars.append(os.path.join(qupath_jars_dir, "libopenslide-jni.so"))

    bfconvert_dir = "./"
    stitcher = TileStitcher(qupath_jars, bfconvert_dir)
    stitcher._start_jvm()

    def teardown():
        try:
            javabridge.kill_vm()
            print("Javabridge vm terminated in teardown")
        except Exception as e:
            print(f"Error during JVM teardown: {e}")
            # f"Error during JVM teardown: {error_message}"

    request.addfinalizer(teardown)

    return stitcher


@pytest.mark.exclude
def test_set_environment_paths(tile_stitcher):
    tile_stitcher.set_environment_paths()
    assert "JAVA_HOME" in os.environ


@pytest.mark.exclude
def test_get_system_java_home(tile_stitcher):
    path = tile_stitcher.get_system_java_home()
    assert isinstance(path, str)


@pytest.mark.exclude
@patch("pathml.preprocessing.tilestitcher.jpype.startJVM")
def test_start_jvm(mocked_jvm, tile_stitcher):
    # Check if JVM was already started
    if jpype.isJVMStarted():
        pytest.skip("JVM was already started, so we skip this test.")
    tile_stitcher._start_jvm()
    mocked_jvm.assert_called()


@pytest.mark.exclude
@patch("pathml.preprocessing.tilestitcher.tifffile")
def test_parse_region(mocked_tifffile, tile_stitcher):
    # Mock the return values
    mocked_tifffile.return_value.__enter__.return_value.pages[
        0
    ].tags.get.side_effect = [
        MagicMock(value=(0, 1)),  # XPosition
        MagicMock(value=(0, 1)),  # YPosition
        MagicMock(value=(1, 1)),  # XResolution
        MagicMock(value=(1, 1)),  # YResolution
        MagicMock(value=100),  # ImageLength
        MagicMock(value=100),  # ImageWidth
    ]
    # filename = "tests/testdata/MISI3542i_M3056_3_Panel1_Scan1_[10530,40933]_component_data.tif"
    filename = "tests/testdata/tilestitching_testdata/MISI3542i_W21-04143_bi016966_M394_OVX_LM_Scan1_[14384,29683]_component_data.tif"
    region = tile_stitcher.parseRegion(filename)
    assert region is not None
    assert isinstance(region, tile_stitcher.ImageRegion)


@pytest.mark.exclude
def test_collect_tif_files(tile_stitcher):
    # Assuming a directory with one tif file for testing
    dir_path = "some_directory"
    os.makedirs(dir_path, exist_ok=True)
    with open(os.path.join(dir_path, "test.tif"), "w") as f:
        f.write("test content")

    files = tile_stitcher._collect_tif_files(dir_path)
    assert len(files) == 1
    assert "test.tif" in files[0]

    os.remove(os.path.join(dir_path, "test.tif"))
    os.rmdir(dir_path)


@pytest.mark.exclude
def test_checkTIFF_valid(tile_stitcher, tmp_path):
    # Create a mock TIFF file
    tiff_path = tmp_path / "mock.tiff"
    tiff_path.write_bytes(b"II*\x00")  # Little-endian TIFF signature
    # assert tile_stitcher.checkTIFF(tiff_path) == True
    assert tile_stitcher.checkTIFF(tiff_path)


@pytest.mark.exclude
def test_checkTIFF_invalid(tile_stitcher, tmp_path):
    # Create a mock non-TIFF file
    txt_path = tmp_path / "mock.txt"
    txt_path.write_text("Not a TIFF file")
    # assert tile_stitcher.checkTIFF(txt_path) == False
    assert not tile_stitcher.checkTIFF(txt_path)


@pytest.mark.exclude
def test_checkTIFF_nonexistent(tile_stitcher):
    # Test with a file that doesn't exist
    with pytest.raises(FileNotFoundError):
        tile_stitcher.checkTIFF("nonexistent_file.tiff")


@pytest.mark.exclude
def test_check_tiff(tile_stitcher):
    valid_tif = b"II*"
    invalid_tif = b"abcd"

    with open("valid_test.tif", "wb") as f:
        f.write(valid_tif)

    with open("invalid_test.tif", "wb") as f:
        f.write(invalid_tif)

    assert tile_stitcher.checkTIFF("tests/testdata/smalltif.tif") is True
    assert tile_stitcher.checkTIFF("invalid_test.tif") is False

    os.remove("valid_test.tif")
    os.remove("invalid_test.tif")


@pytest.mark.exclude
def test_get_outfile_ending_with_ome_tif(tile_stitcher):
    result, result_jpype = tile_stitcher._get_outfile("test.ome.tif")
    assert result == "test.ome.tif"
    assert str(result_jpype) == "test.ome.tif"


@pytest.mark.exclude
def test_get_outfile_without_ending(tile_stitcher):
    result, result_jpype = tile_stitcher._get_outfile("test.ome.tif")
    assert result == "test.ome.tif"
    assert str(result_jpype) == "test.ome.tif"


@pytest.mark.exclude
# Dummy function to "fake" the file download
def mocked_urlretrieve(*args, **kwargs):
    pass


@pytest.mark.exclude
# Mock Zip class as provided
class MockZip:
    def __init__(self, zip_path, *args, **kwargs):
        self.zip_path = zip_path

    def __enter__(self):
        with zipfile.ZipFile(self.zip_path, "w") as zipf:
            zipf.writestr("dummy.txt", "This is dummy file content")
        return self

    def __exit__(self, *args):
        os.remove(self.zip_path)

    def extractall(self, path, *args, **kwargs):
        bftools_dir = os.path.join(path, "bftools")
        if not os.path.exists(bftools_dir):
            os.makedirs(bftools_dir)

        with open(os.path.join(bftools_dir, "bfconvert"), "w") as f:
            f.write("#!/bin/sh\necho 'dummy bfconvert'")

        with open(os.path.join(bftools_dir, "bf.sh"), "w") as f:
            f.write("#!/bin/sh\necho 'dummy bf.sh'")


# Assuming the TileStitcher class definition is available in the current context
# If not, you should import it


@pytest.mark.exclude
def mock_create_zip(zip_path):
    """
    Creates a mock zip file at the given path.
    """
    # Create mock files to add to the ZIP
    with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
        tmpfile.write(b"Mock file content")

    # Create the mock ZIP file
    with zipfile.ZipFile(zip_path, "w") as zipf:
        zipf.write(tmpfile.name, "mock_file.txt")
    os.unlink(tmpfile.name)  # Clean up the temporary file


@pytest.mark.exclude
@pytest.fixture
def bfconvert_dir(tmp_path):
    return tmp_path / "bfconvert_dir"


@pytest.mark.exclude
def test_bfconvert_version_print(tile_stitcher, bfconvert_dir):
    tile_stitcher.setup_bfconvert(bfconvert_dir)
    output = subprocess.check_output([tile_stitcher.bfconvert_path, "-version"])
    assert output.lower().startswith(b"version:")

    # assert subprocess.check_output([tile_stitcher.bfconvert_path, "-version"]) == b'version 1.0.0'


@pytest.mark.exclude
def test_permission_error_on_directory_creation(tile_stitcher):
    with patch("os.makedirs", side_effect=PermissionError("Permission denied")):
        with pytest.raises(PermissionError):
            tile_stitcher.setup_bfconvert("/fake/path")


# @pytest.mark.exclude
# def test_invalid_zip_file(tile_stitcher):
#     with patch("zipfile.ZipFile", side_effect=zipfile.BadZipFile("Invalid ZIP file")):
#         with pytest.raises(zipfile.BadZipFile):
#             tile_stitcher.setup_bfconvert("/fake/path")


@pytest.mark.exclude
def test_permission_error_on_chmod(tile_stitcher):
    with patch("os.chmod", side_effect=PermissionError("Permission denied")):
        with pytest.raises(PermissionError):
            tile_stitcher.setup_bfconvert("/fake/path")


@pytest.mark.exclude
def throwing_function(*args, **kwargs):
    raise Exception("Simulated error")


@pytest.mark.exclude
@pytest.fixture
def mock_tools_dir(tmp_path):
    return tmp_path / "tools"


@pytest.mark.exclude
@pytest.fixture
def mock_zip_path(mock_tools_dir):
    return mock_tools_dir / "bftools.zip"


@pytest.mark.exclude
def mock_urlretrieve(*args, **kwargs):
    with zipfile.ZipFile(args[1], "w") as zipf:
        zipf.writestr("bftools/bfconvert", "dummy content")
        zipf.writestr("bftools/bf.sh", "dummy content")


@pytest.mark.exclude
@patch("urllib.request.urlretrieve", side_effect=mock_urlretrieve)
@patch("os.makedirs", side_effect=PermissionError)
def test_invalid_path(mock_makedirs, mock_urlretrieve, tile_stitcher, mock_tools_dir):
    with pytest.raises(PermissionError):
        tile_stitcher.setup_bfconvert(str(mock_tools_dir))


@pytest.mark.exclude
@patch("urllib.request.urlretrieve", side_effect=mock_urlretrieve)
@patch("zipfile.ZipFile", side_effect=zipfile.BadZipFile)
def test_invalid_zip_file(
    mock_zipfile, mock_urlretrieve, tile_stitcher, mock_tools_dir
):
    with pytest.raises(zipfile.BadZipFile):
        tile_stitcher.setup_bfconvert(str(mock_tools_dir))


@pytest.mark.exclude
@patch("urllib.request.urlretrieve", side_effect=mock_urlretrieve)
@patch("subprocess.check_output", side_effect=subprocess.CalledProcessError(1, "cmd"))
def test_bfconvert_failure(
    mock_subprocess, mock_urlretrieve, tile_stitcher, mock_tools_dir
):
    with pytest.raises(subprocess.CalledProcessError):
        tile_stitcher.setup_bfconvert(str(mock_tools_dir))


@pytest.mark.exclude
def test_is_bfconvert_available_true(tile_stitcher):
    with patch(
        "subprocess.run",
        return_value=subprocess.CompletedProcess(args=[], returncode=0),
    ):
        assert tile_stitcher.is_bfconvert_available() is True


@pytest.mark.exclude
def test_is_bfconvert_available_false(tile_stitcher):
    with patch(
        "subprocess.run",
        return_value=subprocess.CompletedProcess(args=[], returncode=1),
    ):
        assert tile_stitcher.is_bfconvert_available() is False


@pytest.mark.exclude
def test_is_bfconvert_available_file_not_found(tile_stitcher):
    with patch("subprocess.run", side_effect=FileNotFoundError()):
        assert tile_stitcher.is_bfconvert_available() is False


@pytest.mark.exclude
def test_run_bfconvert_bfconvert_not_available(tile_stitcher, capsys):
    tile_stitcher.bfconvert_path = "dummy_path"
    with patch.object(tile_stitcher, "is_bfconvert_available", return_value=False):
        tile_stitcher.run_bfconvert("dummy_stitched_image_path")
        captured = capsys.readouterr()
        assert (
            "bfconvert command not available. Skipping bfconvert step." in captured.out
        )


@pytest.mark.exclude
def test_run_bfconvert_custom_bfconverted_path(tile_stitcher, capsys):
    tile_stitcher.bfconvert_path = "dummy_path"
    with patch.object(tile_stitcher, "is_bfconvert_available", return_value=True):
        with patch("subprocess.run") as mock_run:
            tile_stitcher.run_bfconvert("dummy_stitched_image_path", "custom_path.tif")
            mock_run.assert_called_once_with(
                "./dummy_path -series 0 -separate 'dummy_stitched_image_path' 'custom_path.tif'",
                shell=True,
                check=True,
            )
            captured = capsys.readouterr()
            assert "bfconvert completed. Output file: custom_path.tif" in captured.out


@pytest.mark.exclude
def test_run_bfconvert_default_bfconverted_path(tile_stitcher, capsys):
    tile_stitcher.bfconvert_path = "dummy_path"
    with patch.object(tile_stitcher, "is_bfconvert_available", return_value=True):
        with patch("subprocess.run") as mock_run:
            tile_stitcher.run_bfconvert("dummy_stitched_image_path.tif")
            mock_run.assert_called_once_with(
                "./dummy_path -series 0 -separate 'dummy_stitched_image_path.tif' 'dummy_stitched_image_path_sep.tif'",
                shell=True,
                check=True,
            )
            captured = capsys.readouterr()
            assert (
                "bfconvert completed. Output file: dummy_stitched_image_path_sep.tif"
                in captured.out
            )


@pytest.mark.exclude
def test_run_bfconvert_error(tile_stitcher, capsys):
    tile_stitcher.bfconvert_path = "dummy_path"
    with patch.object(tile_stitcher, "is_bfconvert_available", return_value=True):
        with patch(
            "subprocess.run", side_effect=subprocess.CalledProcessError(1, cmd=[])
        ):
            tile_stitcher.run_bfconvert("dummy_stitched_image_path.tif")
            captured = capsys.readouterr()
            assert "Error running bfconvert command." in captured.out


@pytest.mark.exclude
@pytest.fixture
def sample_files():
    # Paths to your sample TIF files for testing
    return [
        "tests/testdata/tilestitching_testdata/MISI3542i_W21-04143_bi016966_M394_OVX_LM_Scan1_[14384,29683]_component_data.tif"
    ]


@pytest.mark.exclude
def test_integration_stitching(tile_stitcher, sample_files):
    # Mocking the Java object returned by parse_regions
    mocked_java_object = MagicMock()
    with patch.object(tile_stitcher, "parse_regions", return_value=mocked_java_object):
        # Test _collect_tif_files
        collected_files = tile_stitcher._collect_tif_files(sample_files)
        assert set(collected_files) == set(sample_files)

        # Test parse_regions
        regions = tile_stitcher.parse_regions(collected_files)
        assert regions == mocked_java_object

        # Run the actual image stitching on the sample files
        # Assuming the method is `run_image_stitching`
        # NOTE: Adjust the method parameters based on your actual method signature
        tile_stitcher.run_image_stitching(
            sample_files,
            "tests/testdata/tilestitching_testdata/temp",
            separate_series=True,
        )

        # Add more assertions here if you have additional methods or behaviors to verify


@pytest.mark.exclude
def test_write_pyramidal_image_server(tile_stitcher, sample_files):
    infiles = tile_stitcher._collect_tif_files(sample_files)
    fileout, file_jpype = tile_stitcher._get_outfile(
        "tests/testdata/tilestitching_testdata/output_temp"
    )
    downsamples = [1]
    if not infiles or not file_jpype:
        return

    server = tile_stitcher.parse_regions(infiles)
    server = tile_stitcher.ImageServers.pyramidalize(server)
    tile_stitcher._write_pyramidal_image_server(server, file_jpype, downsamples)

    downsamples = None
    tile_stitcher._write_pyramidal_image_server(server, file_jpype, downsamples)
