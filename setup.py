import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

about = {}
# It's also a good practice to specify encoding here, though it may not be necessary if _version.py contains only ASCII characters
with open("pathml/_version.py", "r", encoding="utf-8") as f:
    exec(f.read(), about)

version = about["__version__"]

dependency_links = ["https://download.pytorch.org/whl/cu124"]

setuptools.setup(
    name="pathml",
    version=version,
    author="Jacob Rosenthal, Ryan Carelli et al.",
    author_email="PathML@dfci.harvard.edu",
    description="Tools for computational pathology",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    project_urls={
        "Documentation": "https://pathml.readthedocs.io/en/stable",
        "Source Code": "https://github.com/Dana-Farber-AIOS/pathml",
    },
    dependency_links=dependency_links,
    install_requires=[
        "numpy>=1.26.4,<2",
        "pandas<=2.1.4",
        "scipy>=1.12,<2",
        "scikit-image<=0.22.0",
        "statsmodels",
        "matplotlib>=3.8.0",
        "openslide-python==1.3.1",
        "pydicom==3.0.2",
        "h5py==3.10.0",
        "scikit-learn",
        "dask[distributed]",
        "anndata>=0.7.6,<=0.10.3",
        "scanpy==1.9.6",
        "torch==2.12.0",
        "opencv-contrib-python==4.8.1.78",
        "python-bioformats==4.1.0",
        "python-javabridge==4.0.4",
        "loguru==0.7.2",
        "networkx<=3.2.1",
        "torch-geometric==2.8.0",
        "onnx==1.21.0",
        "onnxruntime>=1.17.0,<1.18",
        "jpype1>=1.5.0",
        "tqdm==4.66.3",
        "onnxscript=0.7.1",
    ],
    classifiers=[
        "License :: OSI Approved :: GNU General Public License v2 (GPLv2)",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Framework :: Sphinx",
        "Framework :: Pytest",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
