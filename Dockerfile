# Stage 1: Java Environment Setup
FROM eclipse-temurin:17-jdk as java-env

# Set JAVA_HOME environment variable
#ENV JAVA_HOME /opt/java/openjdk
RUN echo $JAVA_HOME

# Stage 2: Python Environment Setup with Micromamba
FROM mambaorg/micromamba:bookworm-slim as python-env

# Switch to root user to install packages
USER root

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    openslide-tools g++ gcc libpixman-1-dev libblas-dev liblapack-dev wget \
    libgl1-mesa-dev libgl1-mesa-glx && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create working directory and environment
WORKDIR /app
COPY environment.yml .
RUN micromamba install --yes --name base -f environment.yml && \
    micromamba clean --all --yes

# Copy PathML files
COPY setup.py README.md pyproject.toml /opt/pathml/
COPY pathml/ /opt/pathml/pathml
COPY tests/ /opt/pathml/tests

# Activate the environment
ARG MAMBA_DOCKERFILE_ACTIVATE=1

# Install spams
RUN /opt/conda/bin/pip install spams

# Install PathML and dependencies
RUN /opt/conda/bin/pip install /opt/pathml/

# Test importing pathml to verify installation
RUN /opt/conda/bin/python -c "import pathml"

# Copy Java environment from java-env stage
COPY --from=java-env /opt/java/openjdk /opt/java/openjdk

# Set JAVA_HOME and PATH environment variables
#ENV JAVA_HOME /opt/java/openjdk
#ENV PATH $JAVA_HOME/bin:$PATH

# Set the PATH to include /opt/conda/bin and /opt/conda
ENV PATH /opt/conda/bin:$PATH
ENV PATH /opt/conda:$PATH

# Verify Java installation
RUN java -version

WORKDIR /opt/pathml/
RUN ls
RUN python -m pytest tests/preprocessing_tests/test_tilestitcher.py
RUN pytest -m "not slow and not exclude"

# Set default workdir
WORKDIR /home/pathml


# Expose JupyterLab on port 8888 and set the entrypoint
EXPOSE 8888
ENTRYPOINT ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser"]


# Set the entrypoint to bash for direct shell access
#ENTRYPOINT ["/bin/bash"]
