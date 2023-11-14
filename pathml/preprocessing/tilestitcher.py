"""
Copyright 2021, Dana-Farber Cancer Institute and Weill Cornell Medicine
License: GNU GPL 2.0
"""


import os
import glob
import jpype
import tifffile
import traceback
import sys
import subprocess
import urllib
import zipfile
import platform

class TileStitcher:
    """
    
    This class serves as a Python implementation of a script originally authored by Pete Bankhead, 
    available at https://gist.github.com/petebankhead/b5a86caa333de1fdcff6bdee72a20abe 
    The original script is designed to stitch spectrally unmixed images into a pyramidal OME-TIFF format.

    Make sure QuPath and JDK are installed before using this class.
    
    """

        
    def __init__(self, qupath_jarpath=[], java_path=None, memory="40g", bfconvert_dir='./'):
        """
        Initialize the TileStitcher by setting up the Java Virtual Machine and QuPath environment.
        """
        self.bfconvert_path = self.setup_bfconvert(bfconvert_dir)
        
        if java_path:
            os.environ["JAVA_HOME"] = java_path
        else:
            self.set_environment_paths()
            print('Setting Environment Paths')
        
        # print(qupath_jarpath)
        self.classpath = os.pathsep.join(qupath_jarpath)
        self.memory = memory
        self._start_jvm()
        
    def __del__(self):
        """Ensure the JVM is shutdown when the object is deleted."""
        
        if jpype.isJVMStarted():
            jpype.shutdownJVM()


    def setup_bfconvert(self, bfconvert_dir):
        
        setup_dir = bfconvert_dir
        parent_dir = os.path.dirname(setup_dir)
        tools_dir = os.path.join(parent_dir, 'tools')
        self.bfconvert_path = os.path.join(tools_dir, 'bftools', 'bfconvert')
        self.bf_sh_path = os.path.join(tools_dir, 'bftools', 'bf.sh')
        print(self.bfconvert_path, self.bf_sh_path)

        # Ensure the tools directory exists
        try:
            if not os.path.exists(tools_dir):
                os.makedirs(tools_dir)
        except PermissionError:
            raise PermissionError(f"Permission denied: Cannot create directory {tools_dir}")

        # If bftools folder does not exist, check for bftools.zip or download it
        if not os.path.exists(os.path.join(tools_dir, 'bftools')):
            zip_path = os.path.join(tools_dir, 'bftools.zip')

            if not os.path.exists(zip_path):
                url = 'https://downloads.openmicroscopy.org/bio-formats/latest/artifacts/bftools.zip'
                print(f"Downloading bfconvert from {url}...")
                urllib.request.urlretrieve(url, zip_path)

            print(f"Unzipping {zip_path}...")
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(tools_dir)
            except zipfile.BadZipFile:
                raise zipfile.BadZipFile(f"Invalid ZIP file: {zip_path}")

            if os.path.exists(zip_path):
                os.remove(zip_path)

            print(f"bfconvert set up at {self.bfconvert_path}")
        
        system = platform.system().lower()
        if system == 'linux':
            
            try:
                os.chmod(self.bf_sh_path, os.stat(self.bf_sh_path).st_mode | 0o111)
                os.chmod(self.bfconvert_path, os.stat(self.bfconvert_path).st_mode | 0o111)
            except PermissionError:
                raise PermissionError(f"Permission denied: Cannot chmod files")

        # Print bfconvert version
        try:
            version_output = subprocess.check_output([self.bfconvert_path, "-version"])
            print(f"bfconvert version: {version_output.decode('utf-8').strip()}")
        except subprocess.CalledProcessError:
            raise subprocess.CalledProcessError(1, [self.bfconvert_path, "-version"], output="Failed to get bfconvert version.")

        return self.bfconvert_path
    
    def set_environment_paths(self):
        """
        Set the JAVA_HOME path based on the OS type.
        If the path is not found in the predefined paths dictionary, the function tries
        to automatically find the JAVA_HOME path from the system.
        """
        print('Setting Environment Paths')
        if "JAVA_HOME" in os.environ and os.environ["JAVA_HOME"]:
            # If JAVA_HOME is already set by the user, use that.
            print('Java Home is already set')
            return
        
        # Try to get the JAVA_HOME from the echo command
        java_home = self.get_system_java_home()
        if not java_home:
            raise EnvironmentError("JAVA_HOME not found. Please set it before proceeding or provide it explicitly.")
        
        print(f"Setting Java path to {java_home}")
        os.environ["JAVA_HOME"] = java_home
    
    def get_system_java_home(self):
        """
        Try to automatically find the JAVA_HOME path from the system.
        Return it if found, otherwise return an empty string.
        """
        try:
            # Execute the echo command to get the JAVA_HOME
            java_home = subprocess.getoutput("echo $JAVA_HOME").strip()
            if not java_home:
                raise EnvironmentError("Unable to retrieve JAVA_HOME from the system.")
            return java_home
        except Exception as e:
            print(f"Error retrieving JAVA_HOME from the system: {e}")
            return ""
    
    def run_image_stitching(self, infiles, fileout, downsamples= [1,8], separate_series= False):
        """
        Perform image stitching on the provided TIFF files and output a stitched OME-TIFF image.
        """
        try:
            infiles = self._collect_tif_files(infiles)
            fileout, file_jpype = self._get_outfile(fileout)

            if not infiles or not file_jpype:
                return

            server = self.parse_regions(infiles)
            server = self.ImageServers.pyramidalize(server)
            self._write_pyramidal_image_server(server, file_jpype,downsamples)

            server.close()
            print(f"Image stitching completed. Output file: {file_jpype}")
            
            if separate_series:
                self.run_bfconvert(fileout)

        except Exception as e:
            print(f"Error running image stitching: {e}")
            traceback.print_exc()

            
    def _start_jvm(self):

        """Start the Java Virtual Machine and import necessary QuPath classes."""
        if not jpype.isJVMStarted():
            try:
                # Set memory usage and classpath for the JVM
                memory_usage = f"-Xmx{self.memory}"
                class_path_option = "-Djava.class.path=%s" % self.classpath

                # Try to start the JVM with the specified options
                jpype.startJVM(
                    memory_usage,
                    class_path_option
                )
                
                print(f"Using JVM version: {jpype.getJVMVersion()}")

                # Import necessary QuPath classes
                self._import_qupath_classes()

            except Exception as e:
                # Catch any exception that occurs during JVM startup and print the traceback
                print(f"Error occurred while starting JVM: {e}")
                traceback.print_exc()
                sys.exit(1)
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
            traceback.print_exc()
            
    def run_bfconvert(self, stitched_image_path, bfconverted_path=None):
        if not self.is_bfconvert_available():
            print("bfconvert command not available. Skipping bfconvert step.")
            return
        
        if not bfconverted_path:
            base_path, ext = os.path.splitext(stitched_image_path)
            bfconverted_path = f"{base_path}_sep.tif"
        
        bfconvert_command = f"./{self.bfconvert_path} -series 0 -separate '{stitched_image_path}' '{bfconverted_path}'"
        
        try:
            subprocess.run(bfconvert_command, shell=True, check=True)
            print(f"bfconvert completed. Output file: {bfconverted_path}")
        except subprocess.CalledProcessError:
            print("Error running bfconvert command.")
            
    def is_bfconvert_available(self):
        try:
            result = subprocess.run([f'./{self.bfconvert_path}', "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if result.returncode == 0:
                return True
            else:
                return False
        except FileNotFoundError:
            return False

