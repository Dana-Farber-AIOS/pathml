🤖🔬 **PathML: Tools for computational pathology**

[![Downloads](https://static.pepy.tech/badge/pathml)](https://pepy.tech/project/pathml)
[![Documentation Status](https://readthedocs.org/projects/pathml/badge/?version=latest)](https://pathml.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/Dana-Farber-AIOS/pathml/branch/master/graph/badge.svg?token=UHSQPTM28Y)](https://codecov.io/gh/Dana-Farber-AIOS/pathml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PyPI version](https://img.shields.io/pypi/v/pathml)](https://pypi.org/project/pathml/)
![tests](https://github.com/Dana-Farber-AIOS/pathml/actions/workflows/tests-conda.yml/badge.svg?branch=master)
![dev-tests](https://github.com/Dana-Farber-AIOS/pathml/actions/workflows/tests-conda.yml/badge.svg?branch=dev)

⭐ **PathML objective is to lower the barrier to entry to digital pathology**

Imaging datasets in cancer research are growing exponentially in both quantity and information density. These massive datasets may enable derivation of insights for cancer research and clinical care, but only if researchers are equipped with the tools to leverage advanced computational analysis approaches such as machine learning and artificial intelligence. In this work, we highlight three themes to guide development of such computational tools: scalability, standardization, and ease of use. We then apply these principles to develop PathML, a general-purpose research toolkit for computational pathology. We describe the design of the PathML framework and demonstrate applications in diverse use cases. 

🚀 **The fastest way to get started?**

    docker pull pathml/pathml && docker run -it -p 8888:8888 pathml/pathml

done, what analyses can I write now? 👉 <kbd> <br> **[🔬🤖 Click here to launch your PathML Digital Pathology Assistant 🎓](https://chat.openai.com/g/g-L1IbnIIVt-digital-pathology-assistant-v3-0)** <br> </kbd>

<table> 
<tr>
    <td> <img src="https://github.com/Dana-Farber-AIOS/pathml/assets/25375373/7b1b7293-03cd-4ef1-91d3-8f2efde0899a"/> </td>
    <td>
        
This AI will:
        
- 🤖 write digital pathology analyses for you
- 🔬 walk you through the code, step-by-step
- 🎓 be your teacher, as you embark on your digital pathology journey ❤️

More usage examples [here](./ai-digital-pathology-assistant-v3).
  
</td>
</tr>
</table>



📖 **Official PathML Documentation**

View the official [PathML Documentation on readthedocs](https://pathml.readthedocs.io/en/latest/)

🔥 **Examples! Examples! Examples!**

[↴ Jump to the gallery of examples below](#3-examples)

<br>

<img src=https://raw.githubusercontent.com/Dana-Farber-AIOS/pathml/master/docs/source/_static/images/logo.png width="300"> 

<img src=https://raw.githubusercontent.com/Dana-Farber-AIOS/pathml/master/docs/source/_static/images/overview.png width="750">

# 1. Installation

There are several ways to install `PathML`:

1. `pip install` from PyPI (**recommended for users**)
2. Clone repo to local machine and install from source (recommended for developers/contributors)
3. Use the PathML Docker container
4. Install in Google Colab

Options (1), (2), and (4) require that you first install all external dependencies:
* openslide
* JDK 8

We recommend using conda for environment management. 
Download Miniconda [here](https://docs.conda.io/en/latest/miniconda.html)

## 1.1 Installation option 1: pip install

Create conda environment, this step is common to all platforms (Linux, Mac, Windows):
````
conda create --name pathml python=3.8
conda activate pathml
````

Install external dependencies (for Linux) with [Apt](https://ubuntu.com/server/docs/package-management):
````
sudo apt-get install openslide-tools g++ gcc libblas-dev liblapack-dev
````

Install external dependencies (for MacOS) with [Brew](https://www.brew.sh):
````
brew install openslide
````

Install external dependencies (for Windows) with [vcpkg](https://vcpkg.io/en/):
````
vcpkg install openslide
````

Install [OpenJDK 8](https://openjdk.java.net/), this step is common to all platforms (Linux, Mac, Windows):
````
conda install openjdk==8.0.152
````

Optionally install CUDA (instructions [here](#CUDA))

Install `PathML` from PyPI:
````
pip install pathml
````

## 1.2 Installation option 2: clone repo and install from source

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

Optionally install CUDA (instructions [here](#CUDA))

Install `PathML` from source: 
````
pip install -e .
````

## 1.3 Installation option 3: Docker

First, download or build the PathML Docker container:

![pathml-docker-installation](https://user-images.githubusercontent.com/25375373/191053363-477497a1-9804-48f3-91f9-767dc7f859ed.gif)

- *Step 1:* download PathML container from Docker Hub
   ````
   docker pull pathml/pathml:latest
   ````
  Optionally specify a tag for a particular version, e.g. `docker pull pathml/pathml:2.0.2`. To view possible tags, 
  please refer to the [PathML DockerHub page](https://hub.docker.com/r/pathml/pathml).
  
- *Alternative Step 1* if you have custom hardware: build docker container from source
   ````
   git clone https://github.com/Dana-Farber-AIOS/pathml.git
   cd pathml
   docker build -t pathml/pathml .
   ````

- *Step 2:* Then connect to the container:
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

## 1.4 Installation option 4: Google Colab

To get PathML running in a Colab environment:

````
import os
!pip install openslide-python
!apt-get install openslide-tools
!apt-get install openjdk-8-jdk-headless -qq > /dev/null
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
!update-alternatives --set java /usr/lib/jvm/java-8-openjdk-amd64/jre/bin/java
!java -version
!pip install pathml
````

PathML Tutorials we published in Google Colab
1. [PathML Tutorial Colab #1 - Load an SVS image in PathML and see the image descriptors](https://colab.research.google.com/drive/12ICBsJLCvuubTqb42-Wr5k-2EVDPbbNQ#scrollTo=Qog8Y6wARMgW)
2. [Now that you have PathML installed, all our other examples would work too](https://github.com/Dana-Farber-AIOS/pathml#3-examples) - Only make sure you select an appropriately sized backend or VM in CoLab (i.e., RAM, CPU, Disk, and GPU if necessary) 

*Thanks to all of our open-source collaborators for helping maintain these installation instructions!*  
*Please open an issue for any bugs or other problems during installation process.*

## 1.5 CUDA (optional)

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

# 2. Using with Jupyter (optional)

Jupyter notebooks are a convenient way to work interactively. To use `PathML` in Jupyter notebooks: 

## 2.1 Set JAVA_HOME environment variable

PathML relies on Java to enable support for reading a wide range of file formats.
Before using `PathML` in Jupyter, you may need to manually set the `JAVA_HOME` environment variable 
specifying the path to Java. To do so:

1. Get the path to Java by running `echo $JAVA_HOME` in the terminal in your pathml conda environment (outside of Jupyter)
2. Set that path as the `JAVA_HOME` environment variable in Jupyter:
    ````
    import os
    os.environ["JAVA_HOME"] = "/opt/conda/envs/pathml" # change path as needed
    ````

## 2.2 Register environment as an IPython kernel
````
conda activate pathml
conda install ipykernel
python -m ipykernel install --user --name=pathml
````
This makes the pathml environment available as a kernel in jupyter lab or notebook.

# 3. Examples

Now that you are all set with ``PathML`` installation, let's get started with some analyses you can easily replicate:

<table style="border: 0px !important;">
    <tr>
    <td> 
        
1. [Load over 160+ different types of pathology images using PathML](https://github.com/Dana-Farber-AIOS/pathml/blob/master/examples/loading_images_vignette.ipynb)
2. [H&E Stain Deconvolution and Color Normalization](https://github.com/Dana-Farber-AIOS/pathml/blob/master/examples/stain_normalization.ipynb)
3. [Brightfield imaging pipeline: load an image, preprocess it on a local cluster, and get it read for machine learning analyses in PyTorch](https://github.com/Dana-Farber-AIOS/pathml/blob/master/examples/workflow_HE_vignette.ipynb)
4. [Multiparametric Imaging: Quickstart & single-cell quantification](https://github.com/Dana-Farber-AIOS/pathml/blob/master/examples/multiplex_if.ipynb)
5. [Multiparametric Imaging: CODEX & nuclei quantization](https://github.com/Dana-Farber-AIOS/pathml/blob/master/examples/codex.ipynb)
6. [Train HoVer-Net model to perform nucleus detection and classification, using data from PanNuke dataset](https://github.com/Dana-Farber-AIOS/pathml/blob/master/examples/train_hovernet.ipynb)
7. [Gallery of PathML preprocessing and transformations](https://github.com/Dana-Farber-AIOS/pathml/blob/master/examples/pathml_gallery.ipynb)

</td>                                                                                                                             
        <td>
<img src="https://github.com/Dana-Farber-AIOS/pathml/assets/25375373/502c9e69-e988-4d61-b50f-0d6bfc8af251" width="1000px" />

   </td>
</tr>
</table>


# 4. Citing & known uses

If you use ``PathML`` please cite:

- [**J. Rosenthal et al., "Building tools for machine learning and artificial intelligence in cancer research: best practices and a case study with the PathML toolkit for computational pathology." Molecular Cancer Research, 2022.**](https://doi.org/10.1158/1541-7786.MCR-21-0665)

So far, **PathML** was referenced in 20+ manuscripts:

-   [H. Pakula et al. **Nature Communications**, 2024](https://www.nature.com/articles/s41467-023-44210-1)
-   [B. Ricciuti et al. **Journal of Clinical Oncology**, 2024](https://ascopubs.org/doi/full/10.1200/JCO.23.00580)
-   [A. Song et al. **Nature Reviews Bioengineering**, 2023](https://www.nature.com/articles/s44222-023-00096-8)
-   [I. Virshup et al. **Nature Bioengineering**, 2023](https://www.nature.com/articles/s41587-023-01733-8)
-   [A. Karargyris et al. **Nature Machine Intelligence**, 2023](https://www.nature.com/articles/s42256-023-00652-2)
-   [S. Pati et al. **Nature Communications Engineering**, 2023](https://www.nature.com/articles/s44172-023-00066-3)
-   [C. Gorman et al. **Nature Communications**, 2023](https://www.nature.com/articles/s41467-023-37224-2)
-   [J. Nyman et al. **Cell Reports Medicine**, 2023](https://doi.org/10.1016/j.xcrm.2023.101189)
-   [A. Shmatko et al. **Nature Cancer**, 2022](https://www.nature.com/articles/s43018-022-00436-4)
-   [J. Pocock et al. **Nature Communications Medicine**, 2022](https://www.nature.com/articles/s43856-022-00186-5)
-   [S. Orsulic et al. **Frontiers in Oncology**, 2022](https://www.frontiersin.org/articles/10.3389/fonc.2022.924945/full)
-   [J. Linares et al. **Molecular Cell**, 2021](https://doi.org/10.1016/j.molcel.2021.08.039)
-   the list continues [**here** **🔗**](https://scholar.google.com/scholar?oi=bibs&hl=en&cites=1157052756975292108)

# 5. Users

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

# 6. Contributing

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


# 7. License

The GNU GPL v2 version of PathML is made available via Open Source licensing. 
The user is free to use, modify, and distribute under the terms of the GNU General Public License version 2.

Commercial license options are available also.

# 8. Contact

Questions? Comments? Suggestions? Get in touch!

[pathml@dfci.harvard.edu](mailto:pathml@dfci.harvard.edu)

<img src=https://raw.githubusercontent.com/Dana-Farber-AIOS/pathml/master/docs/source/_static/images/dfci_cornell_joint_logos.png width="750"> 
