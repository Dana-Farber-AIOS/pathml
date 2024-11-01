FROM continuumio/miniconda3

# LABEL about the custom image
LABEL maintainer="PathML@dfci.harvard.edu"
LABEL description="This is custom Docker Image for running PathML"

# install packages on root
USER root

# download and install miniconda and external dependencies
RUN apt-get update && apt-get install -y --no-install-recommends  openslide-tools \
    g++ \
    gcc \
    libpixman-1-0 \
    libblas-dev \
    liblapack-dev 

# download and install opencv dependencies
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# copy pathml files into docker
COPY setup.py README.md /opt/pathml/
COPY requirements/ /opt/pathml/requirements/
COPY examples/ /opt/pathml/examples/
COPY tests/ /opt/pathml/tests
COPY pathml/ /opt/pathml/pathml
COPY docker/entrypoint.sh /opt/pathml/

# make a new conda environment 
RUN conda env create -f /opt/pathml/requirements/environment_docker.yml

# set wording directory
WORKDIR /opt/pathml

# make RUN commands use the new environment
RUN echo "conda activate pathml" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

# install pathml, purest and Jupyter lab
RUN pip3 install /opt/pathml/ pytest
RUN pip3 install jupyter -U && pip3 install jupyterlab

# export port 8888 on the docker
EXPOSE 8888

# run entrypoint script
ENTRYPOINT ["./entrypoint.sh"]