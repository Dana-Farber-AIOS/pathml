import glob
import os
import subprocess
import tempfile
import zipfile
from pathlib import Path
from unittest.mock import mock_open, patch

import javabridge
import jpype
import pytest

from pathml.preprocessing.tilestitcher import TileStitcher
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
def test_jvm_startup(tile_stitcher):

    assert jpype.isJVMStarted(), "JVM should start when TileStitcher is initialized"


@pytest.mark.exclude
@patch("pathml.preprocessing.tilestitcher.tifffile.TiffFile")
@patch("pathml.preprocessing.tilestitcher.TileStitcher.checkTIFF")
def test_parse_region_exception(mocked_check_tiff, mocked_tiff_file, tile_stitcher):
    # Mock the checkTIFF method to always return True
    mocked_check_tiff.return_value = True

    # Mock the TiffFile to raise a FileNotFoundError when used
    mocked_tiff_file.side_effect = FileNotFoundError(
        "Error: File not found dummy_file.tif"
    )
    filename = "dummy_file.tif"

    # Expect FileNotFoundError to be raised
    with pytest.raises(FileNotFoundError) as exc_info:
        tile_stitcher.parseRegion(filename)

    # Assert that the exception message matches what we expect
    assert str(exc_info.value) == "Error: File not found dummy_file.tif"


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
    "qupath_jars, system_type, expected_classpath",
    [
        # Non-Windows paths
        (
            ["/path/to/jar1.jar", "/path/to/jar2.jar"],
            "Linux",
            "-Djava.class.path=/path/to/jar1.jar;/path/to/jar2.jar",
        ),
        (
            ["/path/with multiple/spaces jar.jar", "/another/path.jar"],
            "Linux",
            "-Djava.class.path=/path/with multiple/spaces jar.jar;/another/path.jar",
        ),
        # Windows paths
        (
            ["C:\\path\\to\\jar1.jar", "C:\\path with space\\jar2.jar"],
            "Windows",
            '-Djava.class.path=C:\\path\\to\\jar1.jar;"C:\\path with space\\jar2.jar"',
        ),
        (
            ["C:/mixed/slashes\\jar1.jar", "C:/mixed with space\\slashes jar2.jar"],
            "Windows",
            '-Djava.class.path=C:\\mixed\\slashes\\jar1.jar;"C:\\mixed with space\\slashes jar2.jar"',
        ),
        # Add more path variations if needed
    ],
)
def test_format_jvm_options_classpath(
    qupath_jars, system_type, expected_classpath, tile_stitcher
):
    with patch("platform.system", return_value=system_type):
        # This will simulate the behavior as if the platform is the specified system_type
        result = tile_stitcher.format_jvm_options(qupath_jars, "512m")
        assert result[1] == expected_classpath


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


@pytest.mark.exclude
def test_permission_error_on_directory_creation(tile_stitcher):
    with patch("os.makedirs", side_effect=PermissionError("Permission denied")):
        with pytest.raises(PermissionError):
            tile_stitcher.setup_bfconvert("/fake/path")


@pytest.mark.exclude
@pytest.fixture
def mock_tools_dir(tmp_path):
    return tmp_path / "tools"


@pytest.mark.exclude
def mock_urlretrieve(*args, **kwargs):
    with zipfile.ZipFile(args[1], "w") as zipf:
        zipf.writestr("bftools/bfconvert", "dummy content")
        zipf.writestr("bftools/bf.sh", "dummy content")


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
    stitched_image_path = "dummy_stitched_image_path.ome.tif"

    # Create a dummy stitched image file
    with open(stitched_image_path, "w") as f:
        f.write("dummy content")

    with patch.object(tile_stitcher, "is_bfconvert_available", return_value=True):
        with patch("subprocess.run"):
            with patch("os.remove"):  # Mock os.remove to prevent actual file deletion
                tile_stitcher.run_bfconvert(stitched_image_path, "custom_path.tif")
                captured = capsys.readouterr()
                assert (
                    "bfconvert completed. Output file: custom_path.tif" in captured.out
                )

    # Clean up the dummy file
    os.remove(stitched_image_path)


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


@pytest.mark.exclude
@pytest.fixture
def sample_files():
    # Paths to your sample TIF files for testing
    return [
        "tests/testdata/tilestitching_testdata/MISI3542i_W21-04143_bi016966_M394_OVX_LM_Scan1_[14384,29683]_component_data.tif"
    ]


@pytest.mark.exclude
def test_integration_stitching_exceptions(
    tile_stitcher, sample_files, output_file_path
):
    # Mocking the Java object returned by parse_regions

    tile_stitcher.run_image_stitching(
        sample_files,
        output_file_path,
        downsamples=[],
        separate_series=True,
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
@pytest.mark.skipif(os.name == "nt", reason="chmod not used in windows")
def test_setup_bfconvert_bad_zip_file(tile_stitcher, mock_tools_dir):
    with patch("os.path.exists", return_value=False):
        with patch("urllib.request.urlretrieve"):
            with patch(
                "zipfile.ZipFile", side_effect=zipfile.BadZipFile("Invalid ZIP file")
            ):
                with pytest.raises(zipfile.BadZipFile):
                    tile_stitcher.setup_bfconvert(str(mock_tools_dir))


@pytest.mark.exclude
def test_setup_bfconvert_permission_error_on_chmod(
    tile_stitcher, mock_tools_dir, tmp_path
):
    dummy_bfconvert = tmp_path / "bfconvert"
    dummy_bfconvert.touch()
    dummy_bf_sh = tmp_path / "bf.sh"
    dummy_bf_sh.touch()

    tile_stitcher.bfconvert_path = str(dummy_bfconvert)
    tile_stitcher.bf_sh_path = str(dummy_bf_sh)

    with patch("os.path.exists", return_value=True):
        with patch("os.stat", return_value=os.stat(dummy_bf_sh)):
            with patch("os.chmod", side_effect=PermissionError("Permission denied")):
                with pytest.raises(PermissionError):
                    tile_stitcher.setup_bfconvert(str(mock_tools_dir))


@pytest.mark.exclude
def test_collect_tif_files_invalid_input(tile_stitcher):
    invalid_input = 123  # not a string or list
    result = tile_stitcher._collect_tif_files(invalid_input)
    assert result == []


@pytest.mark.exclude
def test_check_tiff_io_error(tile_stitcher):
    with patch("builtins.open", side_effect=IOError("IO error occurred")):
        with pytest.raises(IOError):
            tile_stitcher.checkTIFF("invalid_file.tif")


@pytest.mark.exclude
def test_start_jvm_exception(tile_stitcher, capsys):
    with patch("jpype.isJVMStarted", return_value=False), patch(
        "jpype.startJVM", side_effect=Exception("JVM start error")
    ):

        tile_stitcher._start_jvm()

        captured = capsys.readouterr()  # Capture the printed output
        assert "Error occurred while starting JVM: JVM start error" in captured.out


@pytest.mark.exclude
def test_import_qupath_classes_exception(tile_stitcher):
    with patch("jpype.JPackage", side_effect=Exception("Import error")):
        with pytest.raises(RuntimeError) as exc_info:
            tile_stitcher._import_qupath_classes()
        assert "Failed to import QuPath classes" in str(exc_info.value)


@pytest.mark.exclude
@patch("builtins.open", mock_open(read_data=b"non TIFF data"))
def test_parse_region_invalid_tiff(tile_stitcher):
    non_tiff_file = "non_tiff_file.txt"
    assert tile_stitcher.parseRegion(non_tiff_file) is None


@pytest.mark.exclude
def test_run_image_stitching_with_empty_input(tile_stitcher, sample_files):
    # Mocking an empty input scenario
    with patch.object(tile_stitcher, "_collect_tif_files", return_value=[]):
        # Output file
        output_file = "output.ome.tif"
        # Running the stitching method
        tile_stitcher.run_image_stitching(sample_files, output_file)


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

    with pytest.raises(EnvironmentError) as excinfo:
        TileStitcher()
    assert "No valid Java path specified, and JAVA_HOME is not set or invalid." in str(
        excinfo.value
    )


@pytest.mark.exclude
@patch("os.path.isdir", return_value=True)
def test_initialization_sets_java_home(mock_isdir):
    mock_java_path = "/mock/path/to/java"
    stitcher = TileStitcher(java_path=mock_java_path)

    assert stitcher.java_home == mock_java_path, "JAVA_HOME not set correctly"
    assert (
        os.environ["JAVA_HOME"] == mock_java_path
    ), "JAVA_HOME environment variable not overridden correctly"
