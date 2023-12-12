"""
Copyright 2021, Dana-Farber Cancer Institute and Weill Cornell Medicine
License: GNU GPL 2.0
"""


import glob
import os
import platform
import subprocess
import traceback
import urllib
import zipfile

import jpype
import tifffile


class TileStitcher:

    """

    This class serves as a Python implementation of a script originally authored by Pete Bankhead,
    available at https://gist.github.com/petebankhead/b5a86caa333de1fdcff6bdee72a20abe
    The original script is designed to stitch spectrally unmixed images into a pyramidal OME-TIFF format.

    Make sure QuPath and JDK are installed before using this class.

    """

    def __init__(
        self, qupath_jarpath=[], java_path=None, memory="40g", bfconvert_dir="./"
    ):
        self.classpath = os.pathsep.join(qupath_jarpath)
        self.memory = memory
        self.bfconvert_dir = bfconvert_dir

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

                jvm_version = jpype.getJVMVersion()
                # Try to start the JVM with the specified options
                jpype.startJVM(memory_usage, class_path_option)

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

    def _collect_tif_files(self, input):
        """Collect .tif files from the input directory or list."""
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
        setup_dir = bfconvert_dir
        parent_dir = os.path.dirname(setup_dir)
        tools_dir = os.path.join(parent_dir, "tools")
        self.bfconvert_path = os.path.join(tools_dir, "bftools", "bfconvert")
        self.bf_sh_path = os.path.join(tools_dir, "bftools", "bf.sh")
        print(self.bfconvert_path, self.bf_sh_path)

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

        # Print bfconvert version
        try:
            version_output = subprocess.check_output([self.bfconvert_path, "-version"])
            print(f"bfconvert version: {version_output.decode('utf-8').strip()}")
        except subprocess.CalledProcessError:
            raise subprocess.CalledProcessError(
                1,
                [self.bfconvert_path, "-version"],
                output="Failed to get bfconvert version.",
            )

        return self.bfconvert_path

    def _get_outfile(self, fileout):
        """Get the output file object for the stitched image."""
        if not fileout.endswith(".ome.tif"):
            fileout += ".ome.tif"
        return fileout, jpype.JClass("java.io.File")(fileout)

    def parseRegion(self, file, z=0, t=0):
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
        return (b1 << 8) + b2

    # Define a function to parse TIFF file metadata and extract the image region
    def parse_regions(self, infiles):
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
        """Convert the parsed image regions into a pyramidal image server and write to file."""
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
        """
        print("Separating Series", separate_series)
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

    def run_bfconvert(self, stitched_image_path, bfconverted_path=None):
        if not self.is_bfconvert_available():
            print("bfconvert command not available. Skipping bfconvert step.")
            return

        if not bfconverted_path:
            base_path = stitched_image_path.rsplit(".ome.tif", 1)[0]
            bfconverted_path = f"{base_path}_separated.tif"

        bfconvert_command = f"./{self.bfconvert_path} -series 0 -separate '{stitched_image_path}' '{bfconverted_path}'"

        # Check if the file already exists and remove it to avoid prompting
        if not os.path.exists(bfconverted_path):

            try:
                subprocess.run(bfconvert_command, shell=True, check=True)
                print(f"bfconvert completed. Output file: {bfconverted_path}")
            except subprocess.CalledProcessError:
                print("Error running bfconvert command.")
        else:

            print("File already exists")

    def is_bfconvert_available(self):
        try:
            result = subprocess.run(
                [f"./{self.bfconvert_path}", "-version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            if result.returncode == 0:
                return True
            else:
                return False
        except FileNotFoundError:
            return False

    def shutdown(self):
        if jpype.isJVMStarted():
            jpype.shutdownJVM()
            print("JVM successfully shutdown")
