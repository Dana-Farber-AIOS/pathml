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
    from pathml.core import HESlide, SlideDataset

    # assuming that all WSIs are in a single directory, all with .svs file extension
    data_dir = Path("/path/to/data/")
    wsi_paths = list(data_dir.glob("*.svs"))

    # create a list of SlideData objects by loading each path
    wsi_list = [HESlide(p) for p in wsi_paths]

    # initialize a SlideDataset
    dataset = SlideDataset(wsi_list)


Supported slide types
---------------------

All slides are represented as :class:`~pathml.core.SlideData` objects.

We provide several convenience classes for loading common types of slides:

.. list-table:: Slide Classes
   :widths: 20 60
   :header-rows: 1

   * - Slide Class
     - Description
   * - :class:`~pathml.core.HESlide`
     - H&E stained images.
   * - :class:`~pathml.core.IHCSlide`
     - IHC stained images
   * - :class:`~pathml.core.MultiparametricSlide`
     - Generic multidimensional, multichannel, time-series images (e.g. multiplexed immunofluorescence).
   * - :class:`~pathml.core.VectraSlide`
     - Multiplex images from Vectra platform.
   * - :class:`~pathml.core.CODEXSlide`
     - Multiplex images from CODEX platform.


It is also possible to load a slide by using the generic :class:`~pathml.core.slide_data.SlideData` class and specifying
explicitly the slide_type and which backend to use (refer to table below):

.. code-block::

    wsi = SlideData("../data/CMU-1.svs", name = "example", slide_backend = "openslide", slide_type = types.HE)

For more information on specifying ``slide_type``, see full documentation at :class:`~pathml.core.SlideType`

Supported file formats
----------------------

Whole-slide images can come in a variety of file formats, depending on the type of image and the scanner used.
``PathML`` has several backends for loading images, enabling support for a wide variety of data formats.
All backends use the same API for interfacing with other parts of ``PathML``. Choose the appropriate backend
for the file format:


.. list-table:: PathML Backends
   :widths: 20 60
   :header-rows: 1

   * - Backend
     - Supported file types
   * - :class:`~pathml.core.slide_backends.OpenSlideBackend`
     - | ``.svs``, ``.tif``, ``.tiff``, ``.bif``, ``.ndpi``, ``.vms``, ``.vmu``, ``.scn``, ``.mrxs``, ``.svslide``
       | `Complete list of file types supported by OpenSlide <https://openslide.org/formats/>`_
   * - :class:`~pathml.core.slide_backends.DICOMBackend`
     - | ``.dcm``, ``.dicom``
       | Digital Imaging and Communications in Medicine (DICOM)
   * - :class:`~pathml.core.slide_backends.BioFormatsBackend`
     - | Supports almost all commonly used file formats, including multiparametric and volumetric TIFF files.
       | ``.1sc``, ``.2fl``, ``.acff``, ``.afi``, ``.afm``, ``.aim``, ``.al3d``, ``.ali``, ``.am``,
       | ``.amiramesh``, ``.apl``, ``.arf``, ``.avi``, ``.bif``, ``.bin``, ``.bip``, ``.bmp``,
       | ``.c01``, ``.cfg``, ``.ch5``, ``.cif``, ``.cr2``, ``.crw``, ``.cxd``, ``.czi``,
       | ``.dat``, ``.dat``, ``.db``, ``.dib``, ``.dm2``, ``.dm3``, ``.dm4``, ``.dti``, ``.dv``,
       | ``.eps``, ``.epsi``, ``.exp``, ``.fdf``, ``.fff``, ``.ffr``, ``.fits``, ``.fli``, ``.frm``,
       | ``.gel``, ``.grey``, ``.hdr``, ``.hdr``, ``.hdr``, ``.hdr``, ``.hed``, ``.his``, ``.htd``,
       | ``.htd``, ``.hx``, ``.i2i``, ``.ics``, ``.ids``, ``.im3``, ``.img``, ``.img``, ``.ims``,
       | ``.inr``, ``.ipl``, ``.ipm``, ``.ipw``, ``.j2k``, ``.jp2``, ``.jpf``, ``.jpk``, ``.jpx``,
       | ``.klb``, ``.l2d``, ``.labels``, ``.lei``, ``.lif``, ``.liff``, ``.lim``, ``.lms``,
       | ``.lsm``, ``.map``, ``.mdb``, ``.mnc``, ``.mng``, ``.mod``, ``.mov``, ``.mrc``,
       | ``.mrcs``, ``.mrw``, ``.msr``, ``.msr``, ``.mtb``, ``.mvd2``, ``.naf``, ``.nd``,
       | ``.nef``, ``.nhdr``, ``.nii``, ``.nii.gz``, ``.nrrd``, ``.obf``, ``.obsep``, ``.oib``,
       | ``.oif``, ``.oir``, ``.ome``, ``.ome.btf``, ``.ome.tf2``, ``.ome.tf8``, ``.ome.tif``,
       | ``.ome.tiff``, ``.ome.xml``, ``.par``, ``.pbm``, ``.pcoraw``, ``.pcx``, ``.pds``,
       | ``.pgm``, ``.pic``, ``.pict``, ``.png``, ``.pnl``, ``.ppm``, ``.pr3``, ``.ps``, ``.psd``,
       | ``.qptiff``, ``.r3d``, ``.raw``, ``.rcpnl``, ``.rec``, ``.rec``, ``.scn``, ``.scn``, ``.sdt``,
       | ``.seq``, ``.sif``, ``.sld``, ``.sld``, ``.sm2``, ``.sm3``, ``.spc``, ``.spe``, ``.spi``,
       | ``.st``, ``.stk``, ``.stk``, ``.stp``, ``.sxm``, ``.tfr``, ``.tga``, ``.tif``, ``.tiff``,
       | ``.tnb``, ``.top``, ``.vff``, ``.vsi``, ``.vws``, ``.wat``, ``.wlz``, ``.wpi``,
       | ``.xdce``, ``.xml``, ``.xqd``, ``.xqf``, ``.xv``, ``.xys``, ``.zfp``, ``.zfr``, ``.zvi``
       | `Complete list of file types supported by Bio-Formats <https://docs.openmicroscopy.org/bio-formats/latest/supported-formats.html>`_
