"""
Copyright 2021, Dana-Farber Cancer Institute and Weill Cornell Medicine
License: GNU GPL 2.0
"""

import glob
import os
import platform
import subprocess
import sys
import urllib
import zipfile
from pathlib import Path

import jpype
import tifffile


class TileStitcher:
    """
    A Python class for stitching tiled images, specifically designed for spectrally unmixed images in a pyramidal OME-TIFF format.

    This class is a Python implementation of Pete Bankhead's script for image stitching, available at available at
    https://gist.github.com/petebankhead/b5a86caa333de1fdcff6bdee72a20abe.
    It requires QuPath and JDK to be installed prior to use.

    Args:
        qupath_jarpath (list): Paths to QuPath JAR files.
        java_path (str): Path to Java installation.
        memory (str): Memory allocation for the JVM.
        bfconvert_dir (str): Directory for Bio-Formats conversion tools.
    """

    def __init__(
        self, qupath_jarpath=[], java_path=None, memory="40g", bfconvert_dir="./"
    ):
        """
        Initialize the TileStitcher class with given parameters and start the JVM.

        Args:
            qupath_jarpath (list): Paths to QuPath JAR files.
            java_path (str, optional): Path to Java installation. Uses JAVA_HOME if not provided.
            memory (str, optional): Memory allocation for JVM. Defaults to "40g".
            bfconvert_dir (str, optional): Directory for Bio-Formats conversion tools. Defaults to "./".
        """

        self.classpath = os.pathsep.join(qupath_jarpath)
        self.memory = memory
        self.bfconvert_dir = bfconvert_dir
        self.shell = sys.platform not in ["linux", "darwin"]
        self.java_home = self._set_java_home(java_path)

        self._start_jvm()

    def __del__(self):
        if jpype.isJVMStarted():
            jpype.shutdownJVM()
            print("JVM successfully shutdown.")

    def _set_java_home(self, java_path):
        if java_path and os.path.isdir(java_path):
            os.environ["JAVA_HOME"] = java_path
            print(f"Java path set and JAVA_HOME overridden to: {java_path}")
            return java_path
        elif "JAVA_HOME" in os.environ and os.path.isdir(os.environ["JAVA_HOME"]):
            return os.environ["JAVA_HOME"]
        else:
            raise JVMInitializationError(
                "No valid Java path specified, and JAVA_HOME is not set or invalid."
            )

    def _start_jvm(self):
        """Start the Java Virtual Machine and import necessary QuPath classes."""
        if not jpype.isJVMStarted():
            try:
                jpype.startJVM(
                    jpype.getDefaultJVMPath(),
                    f"-Xmx{self.memory}",
                    f"-Djava.class.path={self.classpath}",
                    convertStrings=False,
                )
                print(
                    f"Using JVM version: {jpype.getJVMVersion()} from {jpype.getDefaultJVMPath()}"
                )
            except jpype.JVMNotFoundException as e:  # pragma: no cover
                raise JVMInitializationError(f"Failed to find JVM: {e}")
            except Exception as e:
                raise JVMInitializationError(f"Unexpected error starting JVM: {e}")
        else:
            print("JVM was already started.")

        self._import_qupath_classes()

    def _import_qupath_classes(self):
        """Import necessary QuPath classes after starting JVM."""

        try:
            print("Importing required QuPath classes")
            # QuPath class imports
            self.ImageServerProvider = jpype.JPackage(
                "qupath.lib.images.servers"
            ).ImageServerProvider
            self.ImageServers = jpype.JPackage("qupath.lib.images.servers").ImageServers
            self.SparseImageServer = jpype.JPackage(
                "qupath.lib.images.servers"
            ).SparseImageServer
            self.OMEPyramidWriter = jpype.JPackage(
                "qupath.lib.images.writers.ome"
            ).OMEPyramidWriter
            self.ImageRegion = jpype.JPackage("qupath.lib.regions").ImageRegion
            self.ImageIO = jpype.JPackage("javax.imageio").ImageIO
            self.BaselineTIFFTagSet = jpype.JPackage(
                "javax.imageio.plugins.tiff"
            ).BaselineTIFFTagSet
            self.TIFFDirectory = jpype.JPackage(
                "javax.imageio.plugins.tiff"
            ).TIFFDirectory
            self.BufferedImage = jpype.JPackage("java.awt.image").BufferedImage
        except Exception as e:
            raise QuPathClassImportError(f"Failed to import QuPath classes: {e}")

    @staticmethod
    def format_jvm_options(qupath_jars, memory):
        memory_option = f"-Xmx{memory}"
        formatted_classpath = [
            (
                str(Path(path).as_posix())
                if platform.system() != "Windows"
                else str(Path(path))
            )
            for path in qupath_jars
        ]
        class_path_option = "-Djava.class.path=" + os.pathsep.join(formatted_classpath)
        return memory_option, class_path_option

    def _collect_tif_files(self, input):
        """
        Collect .tif files from a given directory path or list.

        Args:
            input (str or list): A directory path or a list of .tif files.

        Returns:
            list: A list of .tif file paths.
        """
        if isinstance(input, str) and os.path.isdir(input):
            return glob.glob(os.path.join(input, "**/*.tif"), recursive=True)
        elif isinstance(input, list):
            valid_files = [file for file in input if file.endswith(".tif")]
            if not valid_files:
                raise FileCollectionError(
                    "No valid .tif files found in the provided list."
                )
            return valid_files
        else:
            raise FileCollectionError(
                f"Invalid input for collecting .tif files: {input}"
            )

    def setup_bfconvert(self, bfconvert_dir):
        """
        Set up Bio-Formats conversion tool (bfconvert) in the given directory.

        Args:
            bfconvert_dir (str): Directory path for setting up bfconvert.

        Returns:
            str: Path to the bfconvert tool.
        """
        tools_dir = Path(bfconvert_dir).parent / "tools"
        bftools_dir = tools_dir / "bftools"

        self.bfconvert_path = Path(bftools_dir) / "bfconvert"
        self.bf_sh_path = Path(tools_dir) / "bftools" / "bf.sh"

        try:
            if not tools_dir.exists():
                tools_dir.mkdir(parents=True, exist_ok=True)

            if not bftools_dir.exists():
                zip_path = tools_dir / "bftools.zip"
                if not zip_path.exists():
                    url = "https://downloads.openmicroscopy.org/bio-formats/latest/artifacts/bftools.zip"
                    urllib.request.urlretrieve(url, zip_path)
                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    zip_ref.extractall(tools_dir)
                zip_path.unlink()

            try:
                system = platform.system().lower()
                if system in ["darwin", "linux"]:

                    os.chmod(self.bf_sh_path, os.stat(self.bf_sh_path).st_mode | 0o111)
                    os.chmod(
                        self.bfconvert_path,
                        os.stat(self.bfconvert_path).st_mode | 0o111,
                    )
            except PermissionError as e:
                raise BFConvertSetupError(
                    f"Permission error on setting executable flag: {e}"
                )

            version_output = subprocess.check_output(
                [str(self.bfconvert_path), "-version"], shell=self.shell
            )
            print(f"bfconvert version: {version_output.decode('utf-8').strip()}")
        except (
            zipfile.BadZipFile,
            PermissionError,
            subprocess.CalledProcessError,
            OSError,
        ) as e:
            raise BFConvertSetupError(f"Error setting up bfconvert: {e}")

        return str(self.bfconvert_path)

    def _get_outfile(self, fileout):
        """
        Prepare the output file for the stitched image.

        Args:
            fileout (str): Path of the output file.

        Returns:
            tuple: A tuple containing the output file path and its Java file object.
        """
        if not fileout.endswith(".ome.tif"):
            fileout += ".ome.tif"
        return fileout, jpype.JClass("java.io.File")(fileout)

    def parseRegion(self, file, z=0, t=0):
        """
        Parse an image region from a given TIFF file.

        Args:
            file (str): Path to the TIFF file.
            z (int, optional): Z-position of the image. Defaults to 0.
            t (int, optional): Time point of the image. Defaults to 0.

        Returns:
            ImageRegion: An ImageRegion object representing the parsed region.
        """
        if not self.checkTIFF(file):
            raise TIFFParsingError(
                f"{file} is not a valid TIFF file"
            )  # pragma: no cover

        try:
            with tifffile.TiffFile(file) as tif:
                tag_xpos, tag_ypos, tag_xres, tag_yres = (
                    tif.pages[0].tags.get("XPosition"),
                    tif.pages[0].tags.get("YPosition"),
                    tif.pages[0].tags.get("XResolution"),
                    tif.pages[0].tags.get("YResolution"),
                )

                if not all([tag_xpos, tag_ypos, tag_xres, tag_yres]):
                    raise TIFFParsingError(f"Required TIFF tags missing for {file}")

                xpos, ypos = (
                    10000 * tag_xpos.value[0] / tag_xpos.value[1],
                    10000 * tag_ypos.value[0] / tag_ypos.value[1],
                )
                xres, yres = (
                    tag_xres.value[0] / (tag_xres.value[1] * 10000),
                    tag_yres.value[0] / (tag_yres.value[1] * 10000),
                )

                x, y, width, height = (
                    int(round(xpos * xres)),
                    int(round(ypos * yres)),
                    tif.pages[0].imagewidth,
                    tif.pages[0].imagelength,
                )

                region = self.ImageRegion.createInstance(x, y, width, height, z, t)
                return region
        except Exception as e:
            raise TIFFParsingError(f"Error parsing TIFF file {file}: {e}")

    def checkTIFF(self, file):
        """
        Check if a given file is a valid TIFF file.

        This method reads the first few bytes of the file to determine if it conforms to TIFF specifications.

        Args:
            file (str): Path to the file to be checked.

        Returns:
            bool: True if the file is a valid TIFF file, False otherwise.
        """

        try:
            with open(file, "rb") as f:
                bytes = f.read(4)
                byteOrder = (bytes[0] << 8) + bytes[1]
                val = (
                    (bytes[3] << 8) + bytes[2]
                    if byteOrder == 0x4949
                    else (bytes[2] << 8) + bytes[3]
                )
                return val == 42 or val == 43
        except Exception as e:
            raise TIFFParsingError(f"Error checking TIFF file {file}: {e}")

    def toShort(self, b1, b2):
        """
        Convert two bytes to a short integer.

        This helper function is used for interpreting the binary data in file headers, particularly for TIFF files.

        Args:
            b1 (byte): The first byte.
            b2 (byte): The second byte.

        Returns:
            int: The short integer represented by the two bytes.
        """

        return (b1 << 8) + b2

    def parse_regions(self, infiles):
        """
        Parse image regions from a list of TIFF files and build a sparse image server.

        Args:
            infiles (list): List of paths to TIFF files.

        Returns:
            SparseImageServer: A server containing the parsed image regions.
        """

        builder = self.SparseImageServer.Builder()
        for f in infiles:
            try:
                region = self.parseRegion(f)
                if region is None:  # pragma: no cover
                    continue  # Skip files that failed to parse without halting the entire operation
                serverBuilder = (
                    self.ImageServerProvider.getPreferredUriImageSupport(
                        self.BufferedImage, jpype.JString(f)
                    )
                    .getBuilders()
                    .get(0)
                )
                builder.jsonRegion(region, 1.0, serverBuilder)
            except Exception as e:  # pragma: no cover
                raise ImageServerConstructionError(
                    f"Error parsing regions from file {f}: {e}"
                )
        return builder.build()

    def _write_pyramidal_image_server(self, server, fileout, downsamples):
        """
        Convert the parsed image regions into a pyramidal image server and write the output to a file.

        Args:
            server (SparseImageServer): The image server containing the stitched image regions.
            fileout (java.io.File): The output file object where the stitched image will be written.
            downsamples (list): A list of downsample levels to use in the pyramidal image server.
        """

        try:
            newOME = self.OMEPyramidWriter.Builder(server)
            if downsamples is None:
                downsamples = server.getPreferredDownsamples()
            newOME.downsamples(downsamples).tileSize(
                512
            ).channelsInterleaved().parallelize().losslessCompression().build().writePyramid(
                fileout.getAbsolutePath()
            )
        except Exception as e:  # pragma: no cover
            raise ImageWritingError(
                f"Error writing pyramidal image server to file {fileout}: {e}"
            )

    def run_image_stitching(
        self, input_dir, output_filename, downsamples=[1, 8], separate_series=False
    ):
        """
        Perform image stitching on the provided TIFF files and output a stitched OME-TIFF image.

        Args:
            input_dir (str): Directory containing the input TIFF files.
            output_filename (str): Filename for the output stitched image.
            downsamples (list, optional): List of downsample levels. Defaults to [1, 8].
            separate_series (bool, optional): Whether to separate the series. Defaults to False.
        """

        try:
            infiles = self._collect_tif_files(input_dir)
            output_file, file_jpype = self._get_outfile(output_filename)

            if not infiles or not file_jpype:
                raise ImageStitchingOperationError(
                    "No input files found or output path is invalid."
                )

            server = self.parse_regions(infiles)
            server = self.ImageServers.pyramidalize(server)
            self._write_pyramidal_image_server(server, file_jpype, downsamples)

            server.close()
            print(f"Image stitching completed. Output file: {output_file}")

            if separate_series:
                self.bfconvert_path = self.setup_bfconvert(self.bfconvert_dir)
                self.run_bfconvert(output_file)
        except Exception as e:
            raise ImageStitchingOperationError(f"Error running image stitching: {e}")

    def run_bfconvert(
        self, stitched_image_path, bfconverted_path=None, delete_original=True
    ):
        """
        Run the Bio-Formats conversion tool on a stitched image.

        Args:
            stitched_image_path (str): Path to the stitched image.
            bfconverted_path (str, optional): Path for the converted image. If None, a default path is generated.
            delete_original (bool): If True, delete the original stitched image after conversion.

        """

        if not self.is_bfconvert_available():
            raise BFConvertExecutionError(
                "bfconvert command not available. Skipping bfconvert step."
            )

        # Generate default bfconverted path if not provided
        bfconverted_path = (
            bfconverted_path
            or f"{stitched_image_path.rsplit('.ome.tif', 1)[0]}_separated.ome.tif"
        )

        # Check if the bfconverted file already exists and remove it to avoid prompting
        bfconverted_file = Path(bfconverted_path)
        if bfconverted_file.exists():
            bfconverted_file.unlink()  # This deletes the file

        try:
            # Execute bfconvert command based on the environment (shell or not)
            if self.shell:
                bfconvert_command = f'"{self.bfconvert_path}" -series 0 -separate "{stitched_image_path}" "{bfconverted_path}"'
                subprocess.run(bfconvert_command, shell=True, check=True)
            else:
                subprocess.run(
                    [
                        self.bfconvert_path,
                        "-series",
                        "0",
                        "-separate",
                        stitched_image_path,
                        bfconverted_path,
                    ],
                    check=True,
                )
            print(f"bfconvert completed. Output file: {bfconverted_path}")
        except subprocess.CalledProcessError as e:
            raise BFConvertExecutionError(
                f"Error running bfconvert command: {e}"
            ) from e

        # Optionally delete the original stitched image
        if delete_original:
            original_file = Path(stitched_image_path)
            if original_file.exists():
                original_file.unlink()
                print(f"Original stitched image deleted: {stitched_image_path}")

    def is_bfconvert_available(self):
        """
        Check if bfconvert is available.
        """

        try:
            result = subprocess.run(
                [self.bfconvert_path, "-version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=self.shell,
            )
            return result.returncode == 0
        except (subprocess.CalledProcessError, FileNotFoundError):
            # raise BFConvertExecutionError("bfconvert tool not found.")
            return False

    def shutdown(self):
        """
        Shut down the Java Virtual Machine (JVM) if it's running.
        """

        if jpype.isJVMStarted():
            jpype.shutdownJVM()
            print("JVM successfully shutdown.")


class TileStitcherError(Exception):
    """General exception for TileStitcher-related errors."""

    pass


class JVMInitializationError(TileStitcherError):
    """Specific exception for errors related to JVM initialization."""

    pass


class QuPathClassImportError(TileStitcherError):
    """Specific exception for errors during QuPath class imports."""

    pass


class BFConvertSetupError(TileStitcherError):
    """Exception raised during the setup or execution of the bfconvert tool."""

    pass


class FileCollectionError(TileStitcherError):
    """Exception raised for errors during file collection."""

    pass


class ImageServerConstructionError(TileStitcherError):
    """Exception for errors during the construction of the image server."""

    pass


class ImageWritingError(TileStitcherError):
    """Exception for errors during writing the pyramidal image server to a file."""

    pass


class ImageStitchingOperationError(TileStitcherError):
    """Exception for errors during the image stitching operation."""

    pass


class TIFFParsingError(TileStitcherError):
    """Exception raised for errors during TIFF parsing."""

    pass


class BFConvertExecutionError(TileStitcherError):
    """Exception raised for errors executing the bfconvert tool."""

    pass


class BFConvertError(TileStitcherError):
    """Exception raised for errors during the bfconvert tool setup or operation."""

    pass
