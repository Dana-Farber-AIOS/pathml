Custom Preprocessing Pipelines
==============================

``PathML`` comes with a set of pre-made pipelines ready to use out of the box.
However, it may also be necessary in many cases to create custom preprocessing pipelines tailored to the specific
application at hand.

Pipeline basics
---------------

Preprocessing pipelines are defined in objects that inherit from the BasePipeline abstract class.
The preprocessing logic for a single slide is defined in the ``run_single()`` method.
Then, when the ``run()`` method is called, the input is checked to see whether it is a single slide or a dataset of
slides. The ``run_single()`` method is then called as appropriate, and multiprocessing is automatically handled in the
case of processing an entire dataset.

To define a new pipeline, all that is necessary is to define the ``run_single()`` method.
The method should take a ``BaseSlide`` object as input (or a specific type of slide inheriting from the ``BaseSlide``
class), and should write the processed output to disk. Because the ``run()`` method is just a wrapper around the
``run_single()`` method, there is no need to override the default ``run()``.

A ``SlideData`` object can be used to hold intermediate outputs, so that a preprocessing step can have access to
outputs from earlier steps.

Interacting with slides
------------------------

Pipelines must take ``BaseSlide`` objects as input.
This interaction between Pipelines and Slides is very important - design choices here can affect pipeline execution
times by orders of magnitude!
This is because whole-slide images can be very large, even exceeding the amount of available memory in most machines!

.. note::

    Naively loading an entire WSI into memory at high-resolution should therefore be avoided in most cases!

Consider these best-practices when designing custom pipelines:

- Make use of the ``BaseSlide.chunks()`` method to process the WSI in smaller chunks
- Perform operations on lower-resolution image levels, when possible (i.e. when the slide has multiple resolutions
  available and the operation will not suffer from decreased resolution)
- Be cognizant of memory requirements at each step in the pipeline
- Avoid loading entire slides into memory at high-resolution!

Using Transforms
-------------------

``PathML`` provides a set of modular Transformation objects to make it easier to define custom preprocessing pipelines.
Individual low-level operations are implemented in ``Transform`` objects, through the ``apply()`` method.
This consistent API makes it convenient to use complex operations in pipelines, and combine them modularly.
There are several types of Transforms, as defined by their inputs and outputs:

================== ========== ===========
Transform type     Input      Output
================== ========== ===========
ImageTransform     image      image
Segmentation       image      mask
MaskTransform      mask       mask
================== ========== ===========

Some things to consider when implementing a custom pipeline:

- Use existing Transforms when possible! This will save time compared to implementing the entire pipeline from scratch.
- If implementing a new transformation or pipeline operation, consider contributing it to ``PathML`` so that other
  users in the community can benefit from your hard work! See: contributing
- Be aware of memory and computation requirements of your pipeline.


Examples
--------

In this example we'll define a Pipeline which reads chunks of the input slide, applies a box blur with a given kernel
size, and then writes the blurred image to disk.

.. code-block::

    import os
    import cv2
    from pathml.preprocessing.base import BasePipeline
    from pathml.preprocessing.transforms import BoxBlur
    from pathml.preprocessing.wsi import HESlide

    class ExamplePipeline(BasePipeline):
        def __init__(self, kernel_size):
            self.kernel_size = kernel_size

        def run_single(self, slide, output_dir):
            blur = BoxBlur(kernel_size)
            for i, chunk in enumerate(slide.chunks(level = 0, size = 1000)):
                blurred_chunk = blur.apply(chunk)
                fname = os.path.join(output_dir, f"chunk{i}.jpg")
                cv2.imwrite(fname, blurred_chunk)

    # usage
    wsi = HESlide("/path/to/wsi.svs")
    ExamplePipeline(kernel_size = 11).run(wsi)


In this example, we define a Transform which changes the order of the channels in the input RGB image.

.. code-block::

    from pathml.preprocessing.base import ImageTransform

    class ChannelSwitch(ImageTransform):
        def apply(self, image):
            # make sure that the input image has 3 channels
            assert image.shape[2] == 3
            out = image
            out[:, :, 0] = image[:, :, 2]
            out[:, :, 1] = image[:, :, 0]
            out[:, :, 2] = image[:, :, 1]
            return out
