🤖🔬 **PathML: Tools for computational pathology**

[![Downloads](https://static.pepy.tech/badge/pathml)](https://pepy.tech/project/pathml)
[![Documentation Status](https://readthedocs.org/projects/pathml/badge/?version=latest)](https://pathml.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/Dana-Farber-AIOS/pathml/branch/master/graph/badge.svg?token=UHSQPTM28Y)](https://codecov.io/gh/Dana-Farber-AIOS/pathml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PyPI version](https://img.shields.io/pypi/v/pathml)](https://pypi.org/project/pathml/)

⭐ **PathML objective is to lower the barrier to entry to digital pathology**

Imaging datasets in cancer research are growing exponentially in both quantity and information density. These massive datasets may enable derivation of insights for cancer research and clinical care, but only if researchers are equipped with the tools to leverage advanced computational analysis approaches such as machine learning and artificial intelligence. In this work, we highlight three themes to guide development of such computational tools: scalability, standardization, and ease of use. We then apply these principles to develop PathML, a general-purpose research toolkit for computational pathology. We describe the design of the PathML framework and demonstrate applications in diverse use cases. 

🚀 **The fastest way to get started?**

    docker pull pathml/pathml && docker run -it -p 8888:8888 pathml/pathml

| Branch | Test status   |
| ------ | ------------- |
| master | ![tests](https://github.com/Dana-Farber-AIOS/pathml/actions/workflows/tests-conda.yml/badge.svg?branch=master) |
| dev    | ![tests](https://github.com/Dana-Farber-AIOS/pathml/actions/workflows/tests-conda.yml/badge.svg?branch=dev) |

<img src=https://raw.githubusercontent.com/Dana-Farber-AIOS/pathml/master/docs/source/_static/images/logo.png width="300"> 

<img src=https://raw.githubusercontent.com/Dana-Farber-AIOS/pathml/master/docs/source/_static/images/overview.png width="750">

**View [documentation](https://pathml.readthedocs.io/en/latest/)**

:construction: the `dev` branch is under active development, with experimental features, bug fixes, and refactors that may happen at any time! 
Stable versions are available as tagged releases on GitHub, or as versioned releases on PyPI


# Installation

There are several ways to install `PathML`:

1. `pip install` from PyPI (**recommended for users**)
2. Clone repo to local machine and install from source (recommended for developers/contributors)
3. Use the PathML Docker container

Options (1) and (2) require that you first install all external dependencies:
* OpenSlide
* JDK 17

We recommend using conda for environment management. 
Download Miniconda [here](https://docs.conda.io/en/latest/miniconda.html)

*Note: these instructions are for Linux. Commands may be different for other platforms.*

### Updating Conda and Using libmamba

Recent versions of Conda have integrated `libmamba`, a faster dependency solver. To benefit from this improvement, first ensure your Conda is updated:

````
conda update -n base conda
````

Then, to install and set the new `libmamba` solver, run:

````
conda install -n base conda-libmamba-solver
conda config --set solver libmamba
````

Please refer to [this Anaconda blog post](https://www.anaconda.com/blog/a-faster-conda-for-a-growing-community).

## Installation option 1: pip install

### Common Steps

Create conda environment:
````
conda create --name pathml python=3.9
conda activate pathml
````

### Platform-Specific External Dependencies

#### Install external dependencies (Linux) with [Apt](https://ubuntu.com/server/docs/package-management):
````
sudo apt-get install openslide-tools g++ gcc libblas-dev liblapack-dev
````

#### Install external dependencies (MacOS) with [Brew](www.brew.sh):
````
brew install openslide
````

#### Windows

##### Option A: Install with [vcpkg](https://vcpkg.io/en/):
````
vcpkg install openslide
````

##### Option B: Using Pre-built OpenSlide Binaries (Alternative)
For Windows users, an alternative to using `vcpkg` is to download and use pre-built OpenSlide binaries. This method is recommended if you prefer a quicker setup.

1. Download the OpenSlide Windows binaries from the [OpenSlide Downloads](https://openslide.org/download/) page.
2. Extract the archive to your desired location, e.g., `C:\OpenSlide\`.

**Importing PathML in Windows:**

Insert the following code snippet at the beginning of your Python script or Jupyter notebook before importing PathML. This code sets up the DLL directory for OpenSlide, ensuring that the library is properly loaded:

```python

# The path can also be read from a config file, etc.
OPENSLIDE_PATH = r'c:\path\to\openslide-win64\bin'

import os
if hasattr(os, 'add_dll_directory'):
    # Windows-specific setup
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    # For other OSes, this step is not needed
    import openslide

# Now you can proceed with using PathML
import pathml

```
This code snippet ensures that the OpenSlide DLLs are correctly found by Python on Windows systems. Replace c:\path\to\openslide-win64\bin with the actual path where you extracted the OpenSlide binaries.

If you encounter any DLL load failures, verify that the OpenSlide `bin` directory is correctly added to your `PATH`.

### Install OpenJDK 17
````
conda install -c conda-forge 'openjdk<=18.0'
````

### Optionally install CUDA

[Follow instructions here](#CUDA)

### Install `PathML` from PyPI
````
pip install pathml
````

## Installation option 2: clone repo and install from source

Clone repo:
````
git clone https://github.com/Dana-Farber-AIOS/pathml.git
cd pathml
````

Create conda environment:

````
conda env create -f environment.yml
conda activate pathml
````
To use GPU acceleration for model training or other tasks, you must install CUDA. The default CUDA version in our environment file is 11.6. To install a different CUDA version, refer to the instructions [here](#CUDA)). 

Install `PathML` from source: 
````
pip install -e .
````

## Installation option 3: Docker

First, download or build the PathML Docker container:

![pathml-docker-installation](https://user-images.githubusercontent.com/25375373/191053363-477497a1-9804-48f3-91f9-767dc7f859ed.gif)

- Option A: download PathML container from Docker Hub
   ````
   docker pull pathml/pathml:latest
   ````
  Optionally specify a tag for a particular version, e.g. `docker pull pathml/pathml:2.0.2`. To view possible tags, 
  please refer to the [PathML DockerHub page](https://hub.docker.com/r/pathml/pathml).
  
- Option B: build docker container from source
   ````
   git clone https://github.com/Dana-Farber-AIOS/pathml.git
   cd pathml
   docker build -t pathml/pathml .
   ````

Then connect to the container:
````
docker run -it -p 8888:8888 pathml/pathml
````

The above command runs the container, which is configured to spin up a jupyter lab session and expose it on port 8888. 
The terminal should display a URL to the jupyter lab session starting with `http://127.0.0.1:8888/lab?token=<.....>`. 
Navigate to that page and you should connect to the jupyter lab session running on the container with the pathml 
environment fully configured. If a password is requested, copy the string of characters following the `token=` in the 
url.

Note that the docker container requires extra configurations to use with GPU.  
Note that these instructions assume that there are no other processes using port 8888.

Please refer to the `Docker run` [documentation](https://docs.docker.com/engine/reference/run/) for further instructions
on accessing the container, e.g. for mounting volumes to access files on a local machine from within the container.

## Option 4: Google Colab

To get PathML running in a Colab environment:

````
import os
!pip install openslide-python
!apt-get install openslide-tools
!apt-get install openjdk-17-jdk-headless -qq > /dev/null
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-17-openjdk-amd64"
!update-alternatives --set java /usr/lib/jvm/java-17-openjdk-amd64/bin/java
!java -version
!pip install pathml
````

*Thanks to all of our open-source collaborators for helping maintain these installation instructions!*  
*Please open an issue for any bugs or other problems during installation process.*

## CUDA

To use GPU acceleration for model training or other tasks, you must install CUDA. 
This guide should work, but for the most up-to-date instructions, refer to the [official PyTorch installation instructions](https://pytorch.org/get-started/locally/).

Check the version of CUDA:
````
nvidia-smi
````

Replace both instances of 'cu116' in `requirements/requirements_torch.txt` with the CUDA version you see. For example, for CUDA 11.7, 'cu116' becomes 'cu117'. 

Then create the environment:

````
conda env create -f environment.yml
conda activate pathml
````

After installing PyTorch, optionally verify successful PyTorch installation with CUDA support: 
````
python -c "import torch; print(torch.cuda.is_available())"
````

# Using with Jupyter

Jupyter notebooks are a convenient way to work interactively. To use `PathML` in Jupyter notebooks: 

## Set JAVA_HOME environment variable

PathML relies on Java to enable support for reading a wide range of file formats.
Before using `PathML` in Jupyter, you may need to manually set the `JAVA_HOME` environment variable 
specifying the path to Java. To do so:

1. Get the path to Java by running `echo $JAVA_HOME` in the terminal in your pathml conda environment (outside of Jupyter)
2. Set that path as the `JAVA_HOME` environment variable in Jupyter:
    ````
    import os
    os.environ["JAVA_HOME"] = "/opt/conda/envs/pathml" # change path as needed
    ````

## Register environment as an IPython kernel
````
conda activate pathml
conda install ipykernel
python -m ipykernel install --user --name=pathml
````
This makes the pathml environment available as a kernel in jupyter lab or notebook.


# Contributing

``PathML`` is an open source project. Consider contributing to benefit the entire community!

There are many ways to contribute to `PathML`, including:

* Submitting bug reports
* Submitting feature requests
* Writing documentation and examples
* Fixing bugs
* Writing code for new features
* Sharing workflows
* Sharing trained model parameters
* Sharing ``PathML`` with colleagues, students, etc.

See [contributing](https://github.com/Dana-Farber-AIOS/pathml/blob/master/CONTRIBUTING.rst) for more details.

# Citing

If you use `PathML` please cite:

- [**J. Rosenthal et al., "Building tools for machine learning and artificial intelligence in cancer research: best practices and a case study with the PathML toolkit for computational pathology." Molecular Cancer Research, 2022.**](https://doi.org/10.1158/1541-7786.MCR-21-0665)

So far, PathML was used in the following manuscripts: 

- [J. Linares et al. **Molecular Cell** 2021](https://www.cell.com/molecular-cell/fulltext/S1097-2765(21)00729-2)
- [A. Shmatko et al. **Nature Cancer** 2022](https://www.nature.com/articles/s43018-022-00436-4)
- [J. Pocock et al. **Nature Communications Medicine** 2022](https://www.nature.com/articles/s43856-022-00186-5)
- [S. Orsulic et al. **Frontiers in Oncology** 2022](https://www.frontiersin.org/articles/10.3389/fonc.2022.924945/full)
- [D. Brundage et al. **arXiv** 2022](https://arxiv.org/abs/2203.13888)
- [A. Marcolini et al. **SoftwareX** 2022](https://www.sciencedirect.com/science/article/pii/S2352711022001558)
- [M. Rahman et al. **Bioengineering** 2022](https://www.mdpi.com/2306-5354/9/8/335)
- [C. Lama et al. **bioRxiv** 2022](https://www.biorxiv.org/content/10.1101/2022.09.28.509751v1.full)
- the list continues [**here 🔗 for 2023 and onwards**](https://scholar.google.com/scholar?oi=bibs&hl=en&cites=1157052756975292108)

# Users

<table style="border: 0px !important;"><tr><td>This is where in the world our most enthusiastic supporters are located:
   <br/><br/>
<img src="https://user-images.githubusercontent.com/25375373/208137141-e450aa86-8433-415a-9cc7-c4274139bdc2.png" width="500px">
   </td><td>   
and this is where they work:
   <br/><br/>
<img src="https://user-images.githubusercontent.com/25375373/208137644-f73c86d0-c5c7-4094-80d9-ea11e0edbdc5.png" width="400px">
</td>                                                                                                                             
</tr>
</table>

Source: https://ossinsight.io/analyze/Dana-Farber-AIOS/pathml#people

# License

The GNU GPL v2 version of PathML is made available via Open Source licensing. 
The user is free to use, modify, and distribute under the terms of the GNU General Public License version 2.

Commercial license options are available also.

# Contact

Questions? Comments? Suggestions? Get in touch!

[pathml@dfci.harvard.edu](mailto:pathml@dfci.harvard.edu)

<img src=https://raw.githubusercontent.com/Dana-Farber-AIOS/pathml/master/docs/source/_static/images/dfci_cornell_joint_logos.png width="750"> 
