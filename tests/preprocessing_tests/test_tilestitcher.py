"""
Copyright 2021, Dana-Farber Cancer Institute and Weill Cornell Medicine
License: GNU GPL 2.0
"""

import glob
import os
import platform
import subprocess
import tempfile
import urllib
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import javabridge
import jpype
import pytest

from pathml.preprocessing.tilestitcher import (
    BFConvertExecutionError,
    BFConvertSetupError,
    FileCollectionError,
    ImageStitchingOperationError,
    ImageWritingError,
    JVMInitializationError,
    QuPathClassImportError,
    TIFFParsingError,
    TileStitcher,
)
from pathml.utils import setup_qupath


@pytest.mark.exclude
@pytest.fixture(scope="module")
def tile_stitcher():
    # Attempt to shutdown JavaBridge-based JVM if running
    try:
        javabridge.kill_vm()
        print("Javabridge vm terminated before starting TileStitcher")
    except Exception as e:
        print(f"No running JavaBridge JVM found: {e}")

    # Setup QuPath
    qupath_home = setup_qupath("../../tools/qupath")
    os.environ["QUPATH_HOME"] = qupath_home
    print(os.environ["JAVA_HOME"])
    # Construct path to QuPath jars
    qupath_jars_dir = os.path.join(qupath_home, "lib", "app")
    qupath_jars = glob.glob(os.path.join(qupath_jars_dir, "*.jar"))
    qupath_jars.append(os.path.join(qupath_jars_dir, "libopenslide-jni.so"))

    bfconvert_dir = "./"
    stitcher = TileStitcher(qupath_jarpath=qupath_jars, bfconvert_dir=bfconvert_dir)

    yield stitcher

    # Teardown code
    try:
        javabridge.kill_vm()
        print("Javabridge vm terminated in teardown")
    except Exception as e:
        print(f"Error during JVM teardown: {e}")


@pytest.mark.exclude
@pytest.fixture(scope="module")
def java_home():
    return os.environ.get("JAVA_HOME")


@pytest.mark.exclude
@pytest.fixture
def output_file_path(tmp_path):
    # tmp_path is a pytest fixture that provides a temporary directory unique to the test invocation
    output_path = tmp_path / "output"
    yield str(output_path)
    # Teardown: Remove the file after the test
    if output_path.exists():
        output_path.unlink()


@pytest.mark.exclude
@pytest.fixture
def mock_jpype_startJVM(monkeypatch):
    monkeypatch.setattr(jpype, "startJVM", MagicMock())


@pytest.mark.exclude
@pytest.fixture
def mock_jpype_shutdownJVM(monkeypatch):
    monkeypatch.setattr(jpype, "shutdownJVM", MagicMock())


@pytest.mark.exclude
def test_jvm_startup(tile_stitcher):

    assert jpype.isJVMStarted(), "JVM should start when TileStitcher is initialized"


@pytest.mark.exclude
def test_jvm_start_with_default_options(tile_stitcher):
    assert jpype.isJVMStarted(), "JVM should have started with default options."


@pytest.mark.exclude
def test_parse_region_exception(tile_stitcher):
    filename = "dummy_file.tif"
    with pytest.raises(TIFFParsingError) as exc_info:
        tile_stitcher.parseRegion(filename)
    assert "Error checking TIFF file dummy_file.tif:" in str(exc_info.value)


@pytest.mark.exclude
@pytest.mark.parametrize(
    "memory, expected_memory_option",
    [
        ("512m", "-Xmx512m"),
        ("1g", "-Xmx1g"),
        ("2048m", "-Xmx2048m"),
        # Add more memory options if needed
    ],
)
def test_format_jvm_options_memory(memory, expected_memory_option, tile_stitcher):
    result = tile_stitcher.format_jvm_options([], memory)
    assert result[0] == expected_memory_option


@pytest.mark.exclude
@pytest.mark.parametrize(
    "qupath_jars, expected_classpath_suffix",
    [
        ([], ""),
        (
            ["/path/to/jar1.jar", "/path/to/jar2.jar"],
            "/path/to/jar1.jar:/path/to/jar2.jar",
        ),
        (
            ["C:\\path\\to\\jar1.jar", "C:\\path\\to\\jar2.jar"],
            "C:\\path\\to\\jar1.jar;C:\\path\\to\\jar2.jar",  # Adjusted to use backslashes and semicolon for Windows
        ),
    ],
)
def test_format_jvm_options_classpath(
    qupath_jars, expected_classpath_suffix, tile_stitcher, monkeypatch
):
    os_name = "nt" if any("C:\\" in jar for jar in qupath_jars) else "posix"
    monkeypatch.setattr(
        platform, "system", lambda: "Windows" if os_name == "nt" else "Linux"
    )
    monkeypatch.setattr(os, "pathsep", ";" if os_name == "nt" else ":")

    _, class_path_option = tile_stitcher.format_jvm_options(qupath_jars, "512m")
    expected_classpath = "-Djava.class.path=" + expected_classpath_suffix

    assert class_path_option == expected_classpath


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
    with pytest.raises(TIFFParsingError):
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
@pytest.fixture(scope="module")
def bfconvert_setup(tile_stitcher, tmp_path_factory):

    bfconvert_dir = tmp_path_factory.mktemp("bfconvert_dir")
    tile_stitcher.setup_bfconvert(str(bfconvert_dir))
    return bfconvert_dir


@pytest.mark.exclude
def test_bfconvert_path_setup(tile_stitcher, bfconvert_setup):
    bfconvert_path = tile_stitcher.setup_bfconvert(str(bfconvert_setup))
    assert str(bfconvert_path) == str(
        tile_stitcher.bfconvert_path
    ), "bfconvert path not set correctly"
    assert os.path.exists(
        bfconvert_path
    ), "bfconvert executable does not exist at the expected path"


@pytest.mark.exclude
def test_bfconvert_version_output(tile_stitcher, bfconvert_setup, capsys):
    tile_stitcher.setup_bfconvert(str(bfconvert_setup))
    captured = capsys.readouterr()
    assert (
        "bfconvert version:" in captured.out
    ), "bfconvert version not printed correctly"


# @pytest.mark.exclude
# def test_permission_error_on_directory_creation(tile_stitcher):
#     with patch("pathlib.Path.mkdir", side_effect=OSError("Simulated directory creation failure")):
#         with pytest.raises(BFConvertSetupError) as exc_info:
#             tile_stitcher.setup_bfconvert("/fake/path")
#         assert "directory creation failure" in str(exc_info.value)


@pytest.mark.exclude
@pytest.fixture
def mock_subprocess(monkeypatch):
    mock = MagicMock()
    monkeypatch.setattr("subprocess.check_output", mock)
    monkeypatch.setattr("subprocess.run", mock)
    return mock


@pytest.mark.exclude
@pytest.fixture
def mock_urlretrieve(monkeypatch):
    def fake_urlretrieve(url, filename):
        # Simulate downloading by creating a dummy zip file at the specified filename
        with zipfile.ZipFile(filename, "w") as zipf:
            zipf.writestr("dummy.txt", "This is dummy content")
            zipf.writestr("bftools/bfconvert", "dummy content")
            zipf.writestr("bftools/bf.sh", "dummy content")

    monkeypatch.setattr(urllib.request, "urlretrieve", fake_urlretrieve)


@pytest.fixture
@pytest.mark.exclude
def mock_tools_dir(tmp_path):
    tools_dir = tmp_path / "tools"
    tools_dir.mkdir()
    return tools_dir


@pytest.mark.exclude
def test_bfconvert_failure(
    mock_subprocess, mock_urlretrieve, tile_stitcher, mock_tools_dir
):
    with patch(
        "subprocess.check_output", side_effect=subprocess.CalledProcessError(1, "cmd")
    ):
        with pytest.raises(BFConvertSetupError):
            tile_stitcher.setup_bfconvert(str(mock_tools_dir))


@pytest.mark.exclude
def test_is_bfconvert_available_file_not_found(tile_stitcher):
    with patch("subprocess.run", side_effect=FileNotFoundError()):
        assert not tile_stitcher.is_bfconvert_available()


@pytest.mark.exclude
def test_run_bfconvert_error(tile_stitcher):
    tile_stitcher.bfconvert_path = "dummy_path"
    with patch.object(
        tile_stitcher, "is_bfconvert_available", return_value=True
    ), patch("subprocess.run", side_effect=subprocess.CalledProcessError(1, "cmd")):
        with pytest.raises(BFConvertExecutionError):
            tile_stitcher.run_bfconvert("dummy_stitched_image_path.tif")


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
@pytest.mark.parametrize(
    "is_available, expected_exception",
    [
        (False, BFConvertExecutionError),
        (True, None),  # No exception should be raised if bfconvert is available
    ],
)
def test_run_bfconvert_bfconvert_not_available(
    tile_stitcher, is_available, expected_exception
):
    tile_stitcher.bfconvert_path = "dummy_path"
    with patch.object(
        tile_stitcher, "is_bfconvert_available", return_value=is_available
    ), patch("subprocess.run"):
        if expected_exception:
            with pytest.raises(expected_exception):
                tile_stitcher.run_bfconvert("dummy_stitched_image_path")
        else:
            tile_stitcher.run_bfconvert("dummy_stitched_image_path")


@pytest.mark.exclude
def test_run_bfconvert_custom_bfconverted_path(tile_stitcher, tmp_path):
    tile_stitcher.bfconvert_path = str(tmp_path / "bfconvert")
    stitched_image_path = str(tmp_path / "dummy_stitched_image_path.ome.tif")
    # Create a dummy stitched image file
    with open(stitched_image_path, "w") as f:
        f.write("dummy content")

    with patch.object(
        tile_stitcher, "is_bfconvert_available", return_value=True
    ), patch("subprocess.run"):
        tile_stitcher.run_bfconvert(
            stitched_image_path, str(tmp_path / "custom_path.tif")
        )


@pytest.mark.exclude
def test_run_bfconvert_success_delete_original(tile_stitcher, capsys):
    tile_stitcher.bfconvert_path = "dummy_path"
    stitched_image_path = "dummy_stitched_image_path.ome.tif"

    # Create a dummy stitched image file
    with open(stitched_image_path, "w") as f:
        f.write("dummy content")

    with patch.object(tile_stitcher, "is_bfconvert_available", return_value=True):
        with patch("subprocess.run"):
            tile_stitcher.run_bfconvert(stitched_image_path, delete_original=True)
            captured = capsys.readouterr()
            assert "bfconvert completed. Output file: " in captured.out
            assert "Original stitched image deleted: " in captured.out

    # Check if the original file was deleted
    assert not os.path.exists(stitched_image_path)


@pytest.mark.exclude
def test_run_bfconvert_no_delete_original(tile_stitcher, capsys):
    tile_stitcher.bfconvert_path = "dummy_path"
    stitched_image_path = "dummy_stitched_image_path.ome.tif"

    # Create a dummy stitched image file
    with open(stitched_image_path, "w") as f:
        f.write("dummy content")

    with patch.object(tile_stitcher, "is_bfconvert_available", return_value=True):
        with patch("subprocess.run"):
            tile_stitcher.run_bfconvert(stitched_image_path, delete_original=False)
            captured = capsys.readouterr()
            assert "bfconvert completed. Output file: " in captured.out
            assert "Original stitched image deleted: " not in captured.out

    # Check if the original file still exists
    assert os.path.exists(stitched_image_path)
    if os.path.exists(stitched_image_path):
        os.unlink(stitched_image_path)


@pytest.mark.exclude
@pytest.fixture
def sample_files():
    # Paths to your sample TIF files for testing
    return [
        "tests/testdata/tilestitching_testdata/MISI3542i_W21-04143_bi016966_M394_OVX_LM_Scan1_[14384,29683]_component_data.tif"
    ]


# @pytest.fixture
# def sample_files(tmp_path):
#     # Create a dummy TIFF file for testing
#     file_path = tmp_path / "dummy_file.tif"
#     file_path.write_text("TIFF data")
#     return [str(file_path)]


@pytest.mark.exclude
def test_integration_stitching_exceptions(
    tile_stitcher, sample_files, output_file_path
):
    with patch.object(
        tile_stitcher,
        "_write_pyramidal_image_server",
        side_effect=Exception("Pyramid error"),
    ):
        with pytest.raises(ImageStitchingOperationError):
            tile_stitcher.run_image_stitching(
                sample_files, output_file_path, downsamples=[], separate_series=True
            )


@pytest.mark.exclude
def test_integration_stitching(tile_stitcher, sample_files, output_file_path):
    # Mocking the Java object returned by parse_regions

    tile_stitcher.run_image_stitching(
        sample_files,
        output_file_path,
        downsamples=[1],
        separate_series=True,
    )


@pytest.mark.exclude
def test_integration_stitching_nodownsamples(tile_stitcher, sample_files, tmp_path):
    # Generate the output file name
    output_file_path = str(tmp_path / "output_temp.ome.tif")
    bfconverted_file_path = output_file_path.rsplit(".ome.tif", 1)[0] + "_separated.tif"

    # Create a dummy bfconverted file to simulate existing file scenario
    with open(bfconverted_file_path, "w") as f:
        f.write("dummy content")

    # Run the stitching process
    tile_stitcher.run_image_stitching(
        sample_files,
        output_file_path,
        downsamples=None,
        separate_series=True,
    )

    # Check if the bfconvert path is set correctly
    assert str(tile_stitcher.bfconvert_path) == str(Path("tools/bftools/bfconvert"))

    # Clean up the dummy file
    os.remove(bfconverted_file_path)


@pytest.mark.exclude
def test_checkTIFF_big_endian(tile_stitcher, tmp_path):
    # Create a mock TIFF file with big-endian byte order
    big_endian_file = tmp_path / "big_endian.tiff"
    big_endian_file.write_bytes(b"MM\x00*")  # Big-endian TIFF signature
    assert tile_stitcher.checkTIFF(big_endian_file)


@pytest.mark.exclude
def test_setup_bfconvert_bad_zip_file(tile_stitcher, mock_tools_dir):
    with patch("urllib.request.urlretrieve"), patch(
        "zipfile.ZipFile", side_effect=zipfile.BadZipFile("Invalid ZIP file")
    ):
        with pytest.raises(BFConvertSetupError):
            tile_stitcher.setup_bfconvert(str(mock_tools_dir))


@pytest.mark.exclude
@pytest.mark.skipif(os.name == "nt", reason="chmod not used in windows")
def test_setup_bfconvert_permission_error_on_chmod(
    tile_stitcher, mock_tools_dir, tmp_path
):
    dummy_bfconvert = tmp_path / "bfconvert"
    dummy_bfconvert.touch()
    dummy_bf_sh = tmp_path / "bf.sh"
    dummy_bf_sh.touch()

    tile_stitcher.bfconvert_path = str(dummy_bfconvert)
    tile_stitcher.bf_sh_path = str(dummy_bf_sh)

    with patch("os.chmod", side_effect=PermissionError("Permission denied")):
        with pytest.raises(BFConvertSetupError):
            tile_stitcher.setup_bfconvert(str(mock_tools_dir))


@pytest.mark.exclude
def test_check_tiff_io_error(tile_stitcher):
    with patch("builtins.open", side_effect=OSError("IO error occurred")):
        with pytest.raises(TIFFParsingError):
            tile_stitcher.checkTIFF("invalid_file.tif")


@pytest.mark.exclude
def test_start_jvm_exception(tile_stitcher, capsys):
    with patch("jpype.isJVMStarted", return_value=False), patch(
        "jpype.startJVM", side_effect=Exception("JVM start error")
    ):
        with pytest.raises(JVMInitializationError):
            tile_stitcher._start_jvm()


@pytest.mark.exclude
def test_import_qupath_classes_exception(tile_stitcher):
    with patch("jpype.JPackage", side_effect=Exception("Import error")):
        with pytest.raises(QuPathClassImportError):
            tile_stitcher._import_qupath_classes()


@pytest.mark.exclude
def test_parse_region_invalid_tiff(tile_stitcher):
    non_tiff_file = "non_tiff_file.txt"
    with pytest.raises(TIFFParsingError):
        tile_stitcher.parseRegion(non_tiff_file)


@pytest.mark.exclude
def test_run_image_stitching_with_empty_input(tile_stitcher, sample_files):
    with patch.object(tile_stitcher, "_collect_tif_files", return_value=[]):
        with pytest.raises(ImageStitchingOperationError):
            tile_stitcher.run_image_stitching(sample_files, "output.ome.tif")


@pytest.mark.exclude
@patch("jpype.JPackage", side_effect=Exception("Import error"))
def test_import_qupath_classes_failure(mock_jpackage, tile_stitcher):
    with pytest.raises(QuPathClassImportError) as exc_info:
        tile_stitcher._import_qupath_classes()
    assert "Failed to import QuPath classes: Import error" in str(exc_info.value)


@pytest.mark.exclude
def test_collect_tif_files_invalid_input(tile_stitcher):
    with pytest.raises(FileCollectionError) as exc_info:
        tile_stitcher._collect_tif_files(12345)  # Invalid input
    assert "Invalid input for collecting .tif files:" in str(exc_info.value)


@pytest.mark.exclude
def test_parse_region_missing_tags(tile_stitcher):

    with pytest.raises(TIFFParsingError) as exc_info:
        tile_stitcher.parseRegion("tests/testdata/smalltif.tif")
    assert "Required TIFF tags missing for" in str(exc_info.value)


@pytest.mark.exclude
@patch("subprocess.run")
@patch(
    "pathml.preprocessing.tilestitcher.TileStitcher.is_bfconvert_available",
    return_value=True,
)
def test_run_bfconvert_success(
    mock_is_bfconvert_available, mock_subprocess_run, tile_stitcher, tmp_path
):
    tile_stitcher.bfconvert_path = "/path/to/bfconvert"
    stitched_image_path = str(tmp_path / "stitched_image.ome.tif")
    bfconverted_path = str(tmp_path / "bfconverted_image.ome.tif")

    # Create a dummy stitched image file to mimic the existence of a file
    Path(stitched_image_path).touch()

    tile_stitcher.run_bfconvert(
        stitched_image_path, bfconverted_path, delete_original=False
    )

    mock_subprocess_run.assert_called()


@pytest.mark.exclude
@patch("subprocess.run")
@patch(
    "pathml.preprocessing.tilestitcher.TileStitcher.is_bfconvert_available",
    return_value=True,
)
def test_run_bfconvert_delete_original(
    mock_is_bfconvert_available, mock_subprocess_run, tile_stitcher, tmp_path
):
    tile_stitcher.bfconvert_path = "/path/to/bfconvert"
    stitched_image_path = str(tmp_path / "stitched_image.ome.tif")

    # Create a dummy stitched image file
    Path(stitched_image_path).touch()

    tile_stitcher.run_bfconvert(stitched_image_path, delete_original=True)

    assert not Path(stitched_image_path).exists(), "Original file should be deleted."


@pytest.mark.exclude
@patch("subprocess.run")
@patch(
    "pathml.preprocessing.tilestitcher.TileStitcher.is_bfconvert_available",
    return_value=True,
)
def test_run_bfconvert_delete_original_print_message(
    mock_is_bfconvert_available, mock_subprocess_run, tile_stitcher, tmp_path, capsys
):
    tile_stitcher.bfconvert_path = "/path/to/bfconvert"
    stitched_image_path = str(tmp_path / "stitched_image.ome.tif")

    # Create a dummy stitched image file
    Path(stitched_image_path).touch()

    tile_stitcher.run_bfconvert(stitched_image_path, delete_original=True)

    captured = capsys.readouterr()
    assert (
        "Original stitched image deleted:" in captured.out
    ), "Expected deletion message not printed."


@pytest.mark.exclude
def test_toShort(tile_stitcher):
    # Example bytes and their expected short value
    test_cases = [
        ((0x01, 0x02), 0x0102),
        ((0xFF, 0xFF), 0xFFFF),
        ((0x00, 0x00), 0x0000),
        ((0xAB, 0xCD), 0xABCD),
    ]

    for (byte1, byte2), expected in test_cases:
        result = tile_stitcher.toShort(byte1, byte2)
        assert (
            result == expected
        ), f"Expected {expected}, got {result} for bytes {byte1}, {byte2}"


@pytest.mark.exclude
def test_collect_tif_files_success(tile_stitcher, tmp_path):
    # Set up a directory with some .tif files
    dir_path = tmp_path / "tif_files"
    dir_path.mkdir()
    (dir_path / "file1.tif").touch()
    (dir_path / "file2.tif").touch()

    expected_files = sorted([str(dir_path / "file1.tif"), str(dir_path / "file2.tif")])
    collected_files = sorted(tile_stitcher._collect_tif_files(str(dir_path)))

    assert (
        collected_files == expected_files
    ), f"Expected {expected_files}, got {collected_files}"


@pytest.mark.exclude
def test_collect_tif_files_no_valid_files(tile_stitcher):
    # A list of file names without .tif extension
    invalid_files = ["file1.jpg", "file2.png", "file3.docx"]

    with pytest.raises(FileCollectionError) as exc_info:
        tile_stitcher._collect_tif_files(invalid_files)

    assert "No valid .tif files found in the provided list." in str(exc_info.value)


@pytest.mark.exclude
@patch(
    "pathml.preprocessing.tilestitcher.TileStitcher._write_pyramidal_image_server",
    side_effect=ImageWritingError("Stitching error"),
)
def test_image_stitching_operation_error(mock_stitching, sample_files, tile_stitcher):

    with pytest.raises(ImageStitchingOperationError) as exc_info:
        # tile_stitcher.run_image_stitching("/path/to/input_dir", "output_filename.tif")
        # Run the stitching process
        tile_stitcher.run_image_stitching(
            sample_files,
            "output_filename.tif",
            downsamples=-1,
            separate_series=True,
        )
    assert "Error running image stitching: Stitching error" in str(exc_info.value)


@pytest.mark.exclude
@patch(
    "pathml.preprocessing.tilestitcher.TileStitcher.is_bfconvert_available",
    return_value=True,
)
@patch(
    "subprocess.run", side_effect=subprocess.CalledProcessError(1, "bfconvert -version")
)
def test_bfconvert_execution_error(mock_is_available, mock_run, tile_stitcher):
    with pytest.raises(BFConvertExecutionError) as exc_info:
        tile_stitcher.run_bfconvert("/path/to/stitched_image.tif")
    assert "Error running bfconvert command:" in str(exc_info.value)


@pytest.mark.exclude
@patch("subprocess.run", side_effect=FileNotFoundError())
def test_bfconvert_not_available(mock_run, tile_stitcher):
    assert (
        not tile_stitcher.is_bfconvert_available()
    ), "bfconvert should not be available."


@pytest.mark.exclude
@patch("jpype.shutdownJVM")
def test_tile_stitcher_shutdown(mocked_shutdown, tile_stitcher):
    tile_stitcher.shutdown()
    mocked_shutdown.assert_called()


@pytest.mark.exclude
def test_initialization_with_java_home_env_var(tile_stitcher, java_home, monkeypatch):
    # valid_java_home = "/path/to/valid/java"  # Example valid path
    monkeypatch.setenv("JAVA_HOME", java_home)

    new_stitcher = TileStitcher()
    assert (
        new_stitcher.java_home == java_home
    ), "JAVA_HOME environment variable not used correctly"


@pytest.mark.exclude
def test_initialization_without_java_path_or_java_home(monkeypatch):
    monkeypatch.delenv("JAVA_HOME", raising=False)  # Remove JAVA_HOME if it exists
    with pytest.raises(JVMInitializationError):
        TileStitcher()


@pytest.mark.exclude
@pytest.fixture
def mock_jpype(monkeypatch):
    # Mock jpype's startJVM and isJVMStarted functions to avoid actual JVM initialization
    monkeypatch.setattr("jpype.startJVM", MagicMock())
    monkeypatch.setattr("jpype.isJVMStarted", MagicMock(return_value=True))
    monkeypatch.setattr(
        "jpype.getDefaultJVMPath", MagicMock(return_value="/mock/path/to/jvm")
    )


@pytest.mark.exclude
def test_initialization_sets_java_home(mock_jpype, monkeypatch):
    mock_java_path = "/mock/path/to/java"
    monkeypatch.setenv("JAVA_HOME", mock_java_path)
    # Ensure jpype.startJVM is not actually called
    with patch("os.path.isdir", return_value=True):
        stitcher = TileStitcher()
        assert stitcher.java_home == mock_java_path, "JAVA_HOME not set correctly"


@pytest.mark.exclude
@patch("jpype.getDefaultJVMPath", side_effect=Exception("JVM path error"))
def test_initialization_with_invalid_java_path(mock_get_jvm_path, tmp_path):
    with pytest.raises(JVMInitializationError):
        TileStitcher(java_path=str(tmp_path))
