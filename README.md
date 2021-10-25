<img src=https://raw.githubusercontent.com/Dana-Farber-AIOS/pathml/master/docs/source/_static/images/logo.png width="300"> 

<img src=https://raw.githubusercontent.com/Dana-Farber-AIOS/pathml/master/docs/source/_static/images/overview.png width="750">

![tests](https://github.com/Dana-Farber-AIOS/pathml/actions/workflows/tests-conda.yml/badge.svg?branch=dev)
[![Documentation Status](https://readthedocs.org/projects/pathml/badge/?version=latest)](https://pathml.readthedocs.io/en/latest/?badge=latest)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PyPI version](https://img.shields.io/pypi/v/pathml)](https://pypi.org/project/pathml/)
[![codecov](https://codecov.io/gh/Dana-Farber-AIOS/pathml/branch/master/graph/badge.svg?token=UHSQPTM28Y)](https://codecov.io/gh/Dana-Farber-AIOS/pathml)

A toolkit for computational pathology and machine learning.

**View [documentation](https://pathml.readthedocs.io/en/latest/)**

**Please cite [our paper](https://www.biorxiv.org/content/10.1101/2021.10.21.465212v1)**

# Installation

There are several ways to install `PathML`:

1. pip install (**recommended for users**)
2. clone repo to local machine and install from source (recommended for developers/contributors)

Options (1) and (2) require that you first install all external dependencies:
* openslide
* JDK 8

We recommend using conda for environment management. 
Download Miniconda [here](https://docs.conda.io/en/latest/miniconda.html)

*Note: these instructions are for Linux. Commands may be different for other platforms.*

## Installation option 1: pip install

Create conda environment
````
conda create --name pathml python=3.8
conda activate pathml
````

Install external dependencies (Linux) with [Apt](https://ubuntu.com/server/docs/package-management)
````
sudo apt-get install openslide-tools g++ gcc libblas-dev liblapack-dev
````

Install external dependencies (MacOS) with [Brew](www.brew.sh)
````
brew install openslide
````

Install [OpenJDK 8](https://openjdk.java.net/)
````
conda install openjdk==8.0.152
````

Optionally install CUDA (instructions [here](#CUDA))

Install `PathML`
````
pip install pathml
````

## Installation option 2: clone repo and install from source

Clone repo
````
git clone https://github.com/Dana-Farber-AIOS/pathml.git
cd pathml
````

Create conda environment
````
conda env create -f environment.yml
conda activate pathml
````

Optionally install CUDA (instructions [here](#CUDA))

Install PathML: 
````
pip install -e .
````

## CUDA

To use GPU acceleration for model training or other tasks, you must install CUDA. 
This guide should work, but for the most up-to-date instructions, refer to the [official PyTorch installation instructions](https://pytorch.org/get-started/locally/).

Check the version of CUDA:
````
nvidia-smi
````

Install correct version of `cudatoolkit`:
````
# update this command with your CUDA version number
conda install cudatoolkit=11.0
````

After installing PyTorch, optionally verify successful PyTorch installation with CUDA support: 
````
python -c "import torch; print(torch.cuda.is_available())"
````

# Using with Jupyter

Jupyter notebooks are a convenient way to work interactively. To use PathML in Jupyter notebooks: 

## Set JAVA_HOME environment variable

PathML relies on Java to enable support for reading a wide range of file formats.
Before using PathML in Jupyter, you may need to manually set the `JAVA_HOME` environment variable 
specifying the path to Java. To do so:

1. Get the path to Java by running `echo $JAVA_HOME` in the terminal in your pathml conda environment (outside of Jupyter)
2. Set that path as the `JAVA_HOME` environment variable in Jupyter:
    ````
    import os
    os.environ["JAVA_HOME"] = "/opt/conda/envs/pathml" # change path as needed
    ````

## Register PathML as an IPython kernel
````
conda activate pathml
conda install ipykernel
python -m ipykernel install --user --name=pathml
````
This makes PathML available as a kernel in jupyter lab or notebook.


# Contributing

``PathML`` is an open source project. Consider contributing to benefit the entire community!

There are many ways to contribute to PathML, including:

* Submitting bug reports
* Submitting feature requests
* Writing documentation and examples
* Fixing bugs
* Writing code for new features
* Sharing workflows
* Sharing trained model parameters
* Sharing ``PathML`` with colleagues, students, etc.

See [contributing](https://github.com/Dana-Farber-AIOS/pathml/blob/master/CONTRIBUTING.rst) for more details.

# License

The GNU GPL v2 version of PathML is made available via Open Source licensing. 
The user is free to use, modify, and distribute under the terms of the GNU General Public License version 2.

Commercial license options are available also.

# Contact

Questions? Comments? Suggestions? Get in touch!

[PathML@dfci.harvard.edu](mailto:PathML@dfci.harvard.edu)

<img src=https://raw.githubusercontent.com/Dana-Farber-AIOS/pathml/master/docs/source/_static/images/dfci_cornell_joint_logos.png width="750"> 
