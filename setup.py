import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

about = {}
with open("pathml/_version.py") as f:
    exec(f.read(), about)
version = about["__version__"]

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
    install_requires=[
        "numpy==1.23.5",
        "pandas<=2.1.4",
        "scipy<=1.11.4",
        "scikit-image<=0.22.0",
        "statsmodels",
        "matplotlib<=3.8.2",
        "openslide-python==1.3.1",
        "pydicom==2.4.4",
        "h5py==3.10.0",
        "scikit-learn",
        "dask[distributed]",
        "anndata>=0.7.6",
        "scanpy==1.9.6",
        "torch==1.13.1",
        "opencv-contrib-python==4.8.1.78",
        "python-bioformats==4.0.7",
        "python-javabridge==4.0.3",
        "loguru==0.7.2",
        "networkx<=3.2.1",
        "torch-geometric==2.3.1",
        "onnx==1.15.0",
        "onnxruntime==1.16.3",
        "jpype1==1.4.1",
        "tqdm==4.66.1",
        "anndata<=0.10.3",
        "pydicom==2.4.4",
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
