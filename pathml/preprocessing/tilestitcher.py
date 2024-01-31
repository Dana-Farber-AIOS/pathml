"""
Copyright 2021, Dana-Farber Cancer Institute and Weill Cornell Medicine
License: GNU GPL 2.0
"""

import glob
import os
import platform
import subprocess
import sys
import traceback
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
            qupath_jarpath (list): List of paths to QuPath JAR files.
            java_path (str, optional): Path to Java installation. If not provided, JAVA_HOME is used.
            memory (str, optional): Memory allocation for JVM. Defaults to "40g".
            bfconvert_dir (str, optional): Directory for Bio-Formats conversion tools. Defaults to "./".
        """

        self.classpath = os.pathsep.join(qupath_jarpath)
        self.memory = memory
        self.bfconvert_dir = bfconvert_dir
        self.shell = sys.platform != "linux"
        if java_path and os.path.isdir(java_path):
            # Override JAVA_HOME with the provided Java path
            os.environ["JAVA_HOME"] = java_path
            self.java_home = java_path
            print(f"Java path set and JAVA_HOME overridden to: {java_path}")
        elif "JAVA_HOME" in os.environ and os.path.isdir(os.environ["JAVA_HOME"]):
            self.java_home = os.environ["JAVA_HOME"]
            print("Using JAVA_HOME from environment variables.")
        else:
            # If neither java_path nor JAVA_HOME is set, raise an error
            raise EnvironmentError(
                "No valid Java path specified, and JAVA_HOME is not set or invalid."
            )

        self._start_jvm()

    def __del__(self):
        if jpype.isJVMStarted():
            jpype.shutdownJVM()
            print("JVM successfully shutdown")

    def _start_jvm(self):
        """Start the Java Virtual Machine and import necessary QuPath classes."""
        if not jpype.isJVMStarted():
            try:
                # Set memory usage and classpath for the JVM
                memory_usage = f"-Xmx{self.memory}"
                class_path_option = f"-Djava.class.path={self.classpath}"

                # Fetch the path to the JVM
                jvm_path = jpype.getDefaultJVMPath()

                # Try to start the JVM with the specified options
                jpype.startJVM(memory_usage, class_path_option)
                jvm_version = jpype.getJVMVersion()

                if jvm_version[0] <= 17:

                    print(
                        "Warning: This Java version might not be fully compatible with some QuPath libraries. Java 17 is recommended."
                    )

                self._import_qupath_classes()

                print(f"Using JVM version: {jvm_version} from {jvm_path}")

            except Exception as e:
                print(f"Error occurred while starting JVM: {e}")
                # sys.exit(1)
            else:
                print("JVM started successfully")
        else:
            print("JVM was already started")

    def _import_qupath_classes(self):
        """Import necessary QuPath classes after starting JVM."""

        try:
            print("Importing required qupath classes")
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
            raise RuntimeError(f"Failed to import QuPath classes: {e}")

    @staticmethod
    def format_jvm_options(qupath_jars, memory):
        # Format the memory setting
        memory_option = f"-Xmx{memory}"

        # Format the classpath
        formatted_classpath = []
        for path in qupath_jars:
            if platform.system() == "Windows":
                # Convert forward slashes to backslashes and wrap paths with spaces in quotes
                path = path.replace("/", "\\")
                if " " in path:
                    path = f'"{path}"'
            formatted_classpath.append(path)

        # Join the classpath entries with a semicolon
        class_path_option = "-Djava.class.path=" + ";".join(formatted_classpath)

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
            return [file for file in input if file.endswith(".tif")]
        else:
            print(
                f"Input must be a directory path or list of .tif files. Received: {input}"
            )
            return []

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
        self.bfconvert_path = bftools_dir / "bfconvert"

        self.bf_sh_path = os.path.join(tools_dir, "bftools", "bf.sh")

        print(self.bfconvert_path, self.bf_sh_path)

        print(
            f"bfconvert_dir: {bfconvert_dir}, tools_dir: {tools_dir}, bfconvert_path: {self.bfconvert_path}"
        )

        # Ensure the tools directory exists
        try:
            if not os.path.exists(tools_dir):
                os.makedirs(tools_dir)
        except PermissionError:
            raise PermissionError(
                f"Permission denied: Cannot create directory {tools_dir}"
            )

        # If bftools folder does not exist, check for bftools.zip or download it
        if not os.path.exists(os.path.join(tools_dir, "bftools")):
            zip_path = os.path.join(tools_dir, "bftools.zip")

            if not os.path.exists(zip_path):
                url = "https://downloads.openmicroscopy.org/bio-formats/latest/artifacts/bftools.zip"
                print(f"Downloading bfconvert from {url}...")
                urllib.request.urlretrieve(url, zip_path)

            print(f"Unzipping {zip_path}...")
            try:
                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    zip_ref.extractall(tools_dir)
            except zipfile.BadZipFile:
                raise zipfile.BadZipFile(f"Invalid ZIP file: {zip_path}")

            if os.path.exists(zip_path):
                os.remove(zip_path)

            print(f"bfconvert set up at {self.bfconvert_path}")

        system = platform.system().lower()
        if system == "linux":
            try:
                os.chmod(self.bf_sh_path, os.stat(self.bf_sh_path).st_mode | 0o111)
                os.chmod(
                    self.bfconvert_path, os.stat(self.bfconvert_path).st_mode | 0o111
                )
            except PermissionError:
                raise PermissionError("Permission denied: Cannot chmod files")

        # try:
        #     version_output = subprocess.check_output([str(self.bfconvert_path), "-version"],shell=True)
        #     print(f"bfconvert version: {version_output.decode('utf-8').strip()}")
        # except subprocess.CalledProcessError as e:
        #     print(f"Failed to get bfconvert version: {e.output.decode()}")

        # Print bfconvert version
        try:

            version_output = subprocess.check_output(
                [str(self.bfconvert_path), "-version"], shell=self.shell
            )
            print(f"bfconvert version: {version_output.decode('utf-8').strip()}")
        except subprocess.CalledProcessError:
            raise subprocess.CalledProcessError(
                1,
                [self.bfconvert_path, "-version"],
                output="Failed to get bfconvert version.",
            )

        return self.bfconvert_path

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

        if self.checkTIFF(file):
            try:
                # Extract the image region coordinates and dimensions from the TIFF tags
                with tifffile.TiffFile(file) as tif:
                    tag_xpos = tif.pages[0].tags.get("XPosition")
                    tag_ypos = tif.pages[0].tags.get("YPosition")
                    tag_xres = tif.pages[0].tags.get("XResolution")
                    tag_yres = tif.pages[0].tags.get("YResolution")
                    if (
                        tag_xpos is None
                        or tag_ypos is None
                        or tag_xres is None
                        or tag_yres is None
                    ):
                        print(f"Could not find required tags for {file}")
                        return None
                    xpos = 10000 * tag_xpos.value[0] / tag_xpos.value[1]
                    xres = tag_xres.value[0] / (tag_xres.value[1] * 10000)
                    ypos = 10000 * tag_ypos.value[0] / tag_ypos.value[1]
                    yres = tag_yres.value[0] / (tag_yres.value[1] * 10000)
                    height = tif.pages[0].tags.get("ImageLength").value
                    width = tif.pages[0].tags.get("ImageWidth").value
                x = int(round(xpos * xres))
                y = int(round(ypos * yres))
                # Create an ImageRegion object representing the extracted image region
                region = self.ImageRegion.createInstance(x, y, width, height, z, t)
                return region
            except Exception as e:
                print(f"Error occurred while parsing {file}: {e}")
                traceback.print_exc()
                raise
        else:
            print(f"{file} is not a valid TIFF file")

    # Define a function to check if a file is a valid TIFF file
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
                byteOrder = self.toShort(bytes[0], bytes[1])
                if byteOrder == 0x4949:  # Little-endian
                    val = self.toShort(bytes[3], bytes[2])
                elif byteOrder == 0x4D4D:  # Big-endian
                    val = self.toShort(bytes[2], bytes[3])
                else:
                    return False
                return val == 42 or val == 43
        except FileNotFoundError:
            print(f"Error: File not found {file}")
            raise FileNotFoundError
        except IOError:
            print(f"Error: Could not open file {file}")
            raise IOError
        except Exception as e:
            print(f"Error: {e}")

    # Define a helper function to convert two bytes to a short integer
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

    # Define a function to parse TIFF file metadata and extract the image region
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
                if region is None:
                    print("WARN: Could not parse region for " + str(f))
                    continue
                serverBuilder = (
                    self.ImageServerProvider.getPreferredUriImageSupport(
                        self.BufferedImage, jpype.JString(f)
                    )
                    .getBuilders()
                    .get(0)
                )
                builder.jsonRegion(region, 1.0, serverBuilder)
            except Exception as e:
                print(f"Error parsing regions from file {f}: {e}")
                traceback.print_exc()
        return builder.build()

    def _write_pyramidal_image_server(self, server, fileout, downsamples):
        """
        Convert the parsed image regions into a pyramidal image server and write the output to a file.

        Args:
            server (SparseImageServer): The image server containing the stitched image regions.
            fileout (java.io.File): The output file object where the stitched image will be written.
            downsamples (list): A list of downsample levels to use in the pyramidal image server.
        """

        # Convert the parsed regions into a pyramidal image server and write to file

        try:
            newOME = self.OMEPyramidWriter.Builder(server)

            # Control downsamples
            if downsamples is None:
                downsamples = server.getPreferredDownsamples()
                print(downsamples)
            newOME.downsamples(downsamples).tileSize(
                512
            ).channelsInterleaved().parallelize().losslessCompression().build().writePyramid(
                fileout.getAbsolutePath()
            )
        except Exception as e:
            print(f"Error writing pyramidal image server to file {fileout}: {e}")
            # traceback.print_exc()
            raise

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
                return

            server = self.parse_regions(infiles)
            server = self.ImageServers.pyramidalize(server)
            self._write_pyramidal_image_server(server, file_jpype, downsamples)

            server.close()
            print(f"Image stitching completed. Output file: {file_jpype}")

            if separate_series:
                print("Separating Series")
                self.bfconvert_path = self.setup_bfconvert(self.bfconvert_dir)
                self.run_bfconvert(output_file)

        except Exception as e:
            print(f"Error running image stitching: {e}")
            traceback.print_exc()

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
            print("bfconvert command not available. Skipping bfconvert step.")
            return

        if not bfconverted_path:
            base_path = stitched_image_path.rsplit(".ome.tif", 1)[0]
            bfconverted_path = f"{base_path}_separated.ome.tif"

        bfconvert_command = f"./{self.bfconvert_path} -series 0 -separate '{stitched_image_path}' '{bfconverted_path}'"
        print("Bfconvert Command: ", bfconvert_command)

        bfconvert_command = [
            str(self.bfconvert_path),
            "-series",
            "0",
            "-separate",
            str(stitched_image_path),
            str(bfconverted_path),
        ]

        # Check if the file already exists and remove it to avoid prompting
        if not os.path.exists(bfconverted_path):
            # Run bfconvert command
            try:
                subprocess.run(bfconvert_command, shell=self.shell, check=True)
                print(f"bfconvert completed. Output file: {bfconverted_path}")

                # Delete the original stitched image if requested
                if delete_original:
                    os.remove(stitched_image_path)
                    print(f"Original stitched image deleted: {stitched_image_path}")

            except subprocess.CalledProcessError:
                print("Error running bfconvert command.")

    def is_bfconvert_available(self):

        try:
            result = subprocess.run(
                [str(self.bfconvert_path), "-version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=self.shell,
            )
            return result.returncode == 0
        except FileNotFoundError:
            return False

    def shutdown(self):
        """
        Shut down the Java Virtual Machine (JVM) if it's running.
        """

        if jpype.isJVMStarted():
            jpype.shutdownJVM()
            print("JVM successfully shutdown")
