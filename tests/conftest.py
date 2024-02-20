"""
Copyright 2021, Dana-Farber Cancer Institute and Weill Cornell Medicine
License: GNU GPL 2.0
"""

# conftest.py
import os
import sys

# Pre-configuration to add the OpenSlide DLL directory to the system's environment variables
import os
import sys

def configure_openslide_path():
    print('Configuring OpenSlide path on', sys.platform)
    
    # Check if the os.add_dll_directory function is available (Python 3.8+ on Windows)
    if hasattr(os, 'add_dll_directory'):
        openslide_path = os.getenv('OPENSLIDE_PATH')
        if openslide_path:
            print('Adding OpenSlide path:', openslide_path)
            os.add_dll_directory(openslide_path)
        else:
            raise RuntimeError("OPENSLIDE_PATH environment variable is not set or incorrect.")
    else:
        print("os.add_dll_directory not available, ensure your Python version is 3.8 or higher on Windows")


# Call the pre-configuration function before importing OpenSlide
configure_openslide_path()

# Now it's safe to import OpenSlide and other dependencies
import cv2
import javabridge
import numpy as np
import openslide
import pytest
import scanpy as sc

from pathml.core import Tile, VectraSlide, types

# Ensure javabridge VM is properly terminated at the end of the session
def pytest_sessionfinish(session, exitstatus):
    javabridge.kill_vm()

@pytest.fixture(autouse=True)
def remove_duplicate_paths():
    yield  # Wait for the test to finish
    # Split the PATH by the OS-specific path separator
    paths = os.environ['PATH'].split(os.pathsep)
    # Remove duplicates while preserving order
    unique_paths = list(dict.fromkeys(paths))
    # Join the unique paths back into a string and set it as the new PATH
    os.environ['PATH'] = os.pathsep.join(unique_paths)
    
# Fixtures and utility functions
def create_HE_tile():
    s = openslide.open_slide("tests/testdata/small_HE.svs")
    im_image = s.read_region(level=0, location=(900, 800), size=(500, 500))
    im_np = np.asarray(im_image)
    im_np_rgb = cv2.cvtColor(im_np, cv2.COLOR_RGBA2RGB)
    masks = np.random.randint(low=1, high=255, size=(im_np_rgb.shape[0], im_np_rgb.shape[1]), dtype=np.uint8)
    masks = {"testmask": masks}
    labs = {
        "test_string_label": "testlabel",
        "test_array_label": np.array([2, 3, 4]),
        "test_int_label": 3,
        "test_float_label": 3.0,
        "test_bool_label": True,
    }
    tile = Tile(image=im_np_rgb, coords=(1, 3), masks=masks, labels=labs)
    return tile

@pytest.fixture
def tile():
    return create_HE_tile()

@pytest.fixture
def tileHE():
    tile = create_HE_tile()
    tile.slide_type = types.HE
    return tile

@pytest.fixture
def tileVectra():
    slidedata = VectraSlide("tests/testdata/small_vectra.qptiff", backend="bioformats")
    region = slidedata.slide.extract_region(location=(0, 0), size=(500, 500))
    masks = np.random.randint(low=1, high=255, size=(region.shape[0], region.shape[1]), dtype=np.uint8)
    masks = {"testmask": masks}
    labs = {
        "test_string_label": "testlabel",
        "test_array_label": np.array([2, 3, 4]),
        "test_int_label": 3,
        "test_float_label": 3.0,
        "test_bool_label": True,
    }
    tile = Tile(image=region, coords=(0, 0), masks=masks, slide_type=types.Vectra, labels=labs)
    return tile

@pytest.fixture
def anndata():
    return sc.datasets.pbmc3k_processed()

@pytest.fixture
def tileCODEX():
    x_dim, y_dim, z_dim, c_dim, t_dim = 100, 100, 3, 4, 5
    codex_image = np.random.rand(x_dim, y_dim, z_dim, c_dim, t_dim)
    tile = Tile(image=codex_image, coords=(0, 0), slide_type=types.CODEX)
    return tile
