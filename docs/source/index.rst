.. PathML documentation master file, created by
   sphinx-quickstart on Mon Apr 13 15:55:15 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to PathML's documentation!
==================================

``PathML`` is a Python package for computational pathology.

``PathML`` is a toolbox to facilitate machine learning workflows for high-resolution whole-slide pathology 
images. This includes modular pipelines for preprocessing, PyTorch DataLoaders for training and benchmarking 
machine learning model performance on standardized datasets, and support for sharing preprocessing pipelines, 
pretrained models, and more.

Development is a collaboration between the AI Operations and Data Science Group in the Department of Informatics 
and Analytics at Dana-Farber Cancer Institute and the Department of Pathology and Laboratory Medicine at Weill 
Cornell Medicine.

.. image:: _static/images/dfci_cornell_joint_logos.png


.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   overview
   installation

.. toctree::
   :maxdepth: 2
   :caption: Preprocessing

   loading_slides
   preprocessing
   custom_pipelines
   h5path
   pipeline_repo

.. toctree::
   :maxdepth: 2
   :caption: Datasets

   datasets

.. toctree::
   :maxdepth: 2
   :caption: Machine Learning

   ml
   models
   model_repo

.. toctree::
   :maxdepth: 2
   :caption: Examples

   examples/link_stain_normalization
   examples/link_nucleus_detection
   examples/link_preprocessing_pipeline

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api_reference_full

.. toctree::
   :maxdepth: 2
   :caption: Contributing

   link_contributing


Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
