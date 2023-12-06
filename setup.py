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
        "pip",
        "numpy>=1.16.4",
        "pandas",
        "scipy",
        "scikit-image",
        "statsmodels",
        "matplotlib",
        "openslide-python",
        "pydicom",
        "h5py",
        "scikit-learn",
        "dask[distributed]",
        "anndata>=0.7.6",
        "scanpy",
        "torch",
        "opencv-contrib-python",
        "python-bioformats>=4.0.0",
        "loguru",
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
