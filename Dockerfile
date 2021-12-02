# syntax = docker/dockerfile:experimental
#
# NOTE: To build this you will need a docker version > 18.06 with
#       experimental enabled and DOCKER_BUILDKIT=1
#
#       If you do not use buildkit you are not going to have a good time
#
#       For reference:
#           https://docs.docker.com/develop/develop-images/build_enhancements/
# This dockerfile creates an environment for using pathml
FROM ubuntu:18.04

ARG PYTHON_VERSION=3.8

# use bash as default shell
SHELL [ "/bin/bash", "--login", "-c" ]

# LABEL about the custom image
LABEL maintainer="PathML@dfci.harvard.edu"
LABEL description="This is custom Docker Image for running PathML."

# Disable Prompt During Packages Installation
ARG DEBIAN_FRONTEND=noninteractive

#install packages on root
USER root

# Update Ubuntu Software repository
RUN apt update

#Path for java runtime
ENV JAVA_HOME="/usr/lib/jvm/java-8-openjdk-amd64/jre/"
ARG JAVA_HOME="/usr/lib/jvm/java-8-openjdk-amd64/jre/"

# download and install external dependencies
RUN apt-get install -y --no-install-recommends  openslide-tools \    
    g++ \     
    gcc \    
    libblas-dev \    
    liblapack-dev \    
    wget \
    curl \
    openjdk-8-jre \    
    openjdk-8-jdk \     
    && rm -rf /var/lib/apt/lists/*

# set up python and conda. based on PyTorch dockerfile:
# https://github.com/pytorch/pytorch/blob/7342b654a1570a4685238f8699d2558040a39ece/Dockerfile#L29-L36
ENV CONDA_DIR /opt/conda
RUN curl -fsSL -v -o ~/miniconda.sh -O  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p $CONDA_DIR && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda install -y python=${PYTHON_VERSION} conda-build pyyaml numpy ipython&& \
    /opt/conda/bin/conda clean -ya
ENV PATH $CONDA_DIR/bin:$PATH

# make conda activate command available from /bin/bash --login shells
RUN echo ". $CONDA_DIR/etc/profile.d/conda.sh" >> ~/.profile
# make conda activate command available from /bin/bash --interactive shells
RUN conda init bash

# create a project directory inside user home
# ENV PROJECT_DIR $HOME/pathml
# RUN mkdir $PROJECT_DIR
# WORKDIR $PROJECT_DIR

# copy environment file into docker
COPY environment.yml environment.yml

# set up environment
# from: https://towardsdatascience.com/conda-pip-and-docker-ftw-d64fe638dc45
# build the conda environment
ENV ENV_PREFIX $PWD/env
RUN conda update --name base --channel defaults conda && \
    conda env create --prefix $ENV_PREFIX --file environment.yml --force && \
    conda clean --all --yes

#install pathml into the environment
COPY pathml/ setup.py ./
RUN conda activate $ENV_PREFIX && pip3 install . && conda deactivate

ENTRYPOINT [ "/usr/local/bin/docker_entrypoint.sh" ]
