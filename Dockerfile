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
FROM quay.io/biocontainers/python-bioformats:4.0.5--pyh5e36f6f_0

# LABEL about the custom image
LABEL maintainer="PathML@dfci.harvard.edu"
LABEL description="This is custom Docker Image for running PathML"

# Disable Prompt During Packages Installation
ARG DEBIAN_FRONTEND=noninteractive

#install packages on root
USER root

# download and install external dependencies
# openjdk8 installed following instructions from:
#   https://linuxize.com/post/install-java-on-debian-10/#installing-openjdk-8
RUN apt-get update && apt-get install -y --no-install-recommends \
    openslide-tools \
    g++ \     
    gcc \    
    libblas-dev \    
    liblapack-dev \
    apt-transport-https \
    ca-certificates \
    wget \
    dirmngr \
    gnupg \
    python3-opencv \
    libgl1 \
    && rm -rf /var/lib/apt/lists/* \
    && pip3 install --upgrade pip

#Path for java runtime
ENV JAVA_HOME="/usr/lib/jvm/adoptopenjdk-8-hotspot-amd64/jre"
ARG JAVA_HOME="/usr/lib/jvm/adoptopenjdk-8-hotspot-amd64/jre"

# copy pathml files into docker
COPY setup.py README.md /opt/pathml/
COPY pathml/ /opt/pathml/pathml

# install pathml
RUN pip3 install /opt/pathml/

WORKDIR /home/pathml

# set up jupyter lab
RUN pip3 install jupyter -U && pip3 install jupyterlab
EXPOSE 8888
ENTRYPOINT ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser"]
