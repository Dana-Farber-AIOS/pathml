import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pathml",
    version="1.0.0",
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
        "pip==21.2.1",
        "numpy>=1.16.4,<1.19.0",
        "pandas==1.1.5",
        "scipy==1.5.4",
        "scikit-image",
        "statsmodels==0.12.2",
        "matplotlib==3.1.3",
        "openslide-python==1.1.2",
        "pydicom==2.1.2",
        "h5py==3.1.0",
        "spams==2.6.2.5",
        "scikit-learn==0.24.2",
        "dask[distributed]==2021.7.1",
        "anndata==0.7.4",
        "scanpy==1.7.2",
        "pre-commit==2.13.0",
        "torch==1.9.0",
        "opencv-contrib-python==4.5.3.56",
        "tensorly==0.6.0",
        "python-bioformats==4.0.0",
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
        "Topic :: Software Development :: Libraries :: Python Modules"
    ],
)
