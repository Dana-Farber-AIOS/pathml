Loading Images
==============

Individual Images
-----------------

The first step in any computational pathology workflow is to load the image from disk.
In ``PathML`` this can be done in one line:

.. code-block::

    wsi = HESlide("../data/CMU-1.svs", name = "example")

Datasets of Images
------------------

Using "in-house" datasets from the local filesystem is also supported.

Simply initialize a :class:`~pathml.core.slide_dataset.SlideDataset` object by passing a list of
individual :class:`~pathml.core.slide_dataset.SlideData` objects:

.. code-block::

    from pathlib import Path
    from pathml.core.slide_dataset import SlideDataset
    from pathml.core.slide_data import HESlide

    # assuming that all WSIs are in a single directory, all with .svs file extension
    data_dir = Path("/path/to/data/")
    wsi_paths = list(data_dir.glob("*.svs"))

    # create a list of SlideData objects by loading each path
    wsi_list = [HESlide(p) for p in wsi_paths]

    # initialize a SlideDataset
    dataset = SlideDataset(wsi_list)


Supported slide types
---------------------

``PathML`` provides tools to load common medical image filetypes including proprietary formats from slide scanners.

.. list-table:: Slide Classes
   :widths: 20 60
   :header-rows: 1

   * - Slide Class
     - Description
   * - :class:`~pathml.core.slide_classes.RGBSlide`
     - Any image that is in RGB.
   * - :class:`~pathml.core.slide_classes.HESlide`
     - H&E stained images.
   * - :class:`~pathml.core.slide_classes.MultiparametricSlide`
     - Multidimensional, multichannel, time-series images (e.g. multiplexed immunofluorescence). 


In general it is recommended to use the pre-made classes for convenience. They implement a hierarchical
class structure which is used internally for some operations. (e.g. H&E slides are a subclass of RGB slides).

It is also possible to load a slide by using the generic :class:`~pathml.core.slide_data.SlideData` class and specifying
explicitly which backend to use (see table below):

.. code-block::

    wsi = SlideData("../data/CMU-1.svs", name = "example", slide_backend = OpenSlideBackend)

Supported file formats
----------------------

Whole-slide images can come in a variety of file formats, depending on the type of image and the scanner used.
``PathML`` has several backends for loading images, enabling support for a wide variety of data formats.


.. list-table:: PathML Backends
   :widths: 20 60
   :header-rows: 1

   * - Backend
     - Supported file types
   * - :class:`~pathml.core.slide_backends.OpenSlideBackend`
     - | ``.svs``, ``.tif``, ``.tiff``, ``.bif``, ``.ndpi``, ``.vms``, ``.vmu``, ``.scn``, ``.mrxs``, ``.svslide``
       | `Complete list of file types supported by OpenSlide <https://openslide.org/formats/>`_
   * - :class:`~pathml.core.slide_backends.DICOMBackend`
     - | ``.dcm``
       | [work in progress]
   * - :class:`~pathml.core.slide_backends.BioFormatsBackend`
     - | Supports almost all commonly used file formats, including multiparametric and volumetric TIFF files.
       | `Complete list of file types supported by Bio-Formats <https://docs.openmicroscopy.org/bio-formats/latest/supported-formats.html>`_
