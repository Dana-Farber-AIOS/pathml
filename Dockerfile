# syntax=docker/dockerfile:1

FROM ubuntu:20.04
# LABEL about the custom image
LABEL maintainer="PathML@dfci.harvard.edu"
LABEL description="This is custom Docker Image for running PathML"

# Disable Prompt During Packages Installation
ARG DEBIAN_FRONTEND=noninteractive

#Set miniconda path
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

ENV JAVA_HOME="/usr/lib/jvm/java-8-openjdk-amd64/jre/"
ARG JAVA_HOME="/usr/lib/jvm/java-8-openjdk-amd64/jre/"

ENV SHELL="/bin/bash"

#install packages on root
USER root

#download and install miniconda and external dependencies
RUN apt-get update && apt-get install -y --no-install-recommends  openslide-tools \
    g++ \
    gcc \
    libpixman-1-0 \
    libblas-dev \
    liblapack-dev \
    wget \
    openjdk-8-jre \
    openjdk-8-jdk \
    && wget \
    https://repo.anaconda.com/miniconda/Miniconda3-py38_4.10.3-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-py38_4.10.3-Linux-x86_64.sh -b \
    && rm -f Miniconda3-py38_4.10.3-Linux-x86_64.sh \
    && rm -rf /var/lib/apt/lists/*

# copy pathml files into docker
COPY setup.py README.md /opt/pathml/
COPY pathml/ /opt/pathml/pathml
COPY tests/ /opt/pathml/tests

# install pathml and deepcell
RUN pip3 install --upgrade pip && pip3 install numpy==1.19.5 && pip3 install python-bioformats==4.0.0 deepcell /opt/pathml/

WORKDIR /home/pathml

# set up jupyter lab
RUN pip3 install jupyter -U && pip3 install jupyterlab
EXPOSE 8888
ENTRYPOINT ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser"]
