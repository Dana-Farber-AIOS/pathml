Custom Preprocessing Pipelines
==============================

``PathML`` makes designing preprocessing pipelines easy. In this section we will walk through how to define a
:class:`~pathml.preprocessing.pipeline.Pipeline` object by composing pre-made
:class:`~pathml.preprocessing.transforms.Transform`s, and how to implement a
new custom :class:`~pathml.preprocessing.transforms.Transform`.

Pipeline basics
---------------

Preprocessing pipelines are defined in :class:`~pathml.preprocessing.pipeline.Pipeline` objects.
When :meth:`~pathml.core.slide_data.SlideData.run`
is called, tiles are lazily extracted from the slide by
:meth:`~pathml.core.slide_data.SlideData.generate_tiles` and passed to the
:class:`~pathml.preprocessing.pipeline.Pipeline`, which modifies the :class:`~pathml.core.tile.Tile` object in place.
Finally, the processed tile is saved.
This design facilitates preprocessing of gigapixel-scale whole-slide images, because :class:`~pathml.core.tile.Tile`
objects are small enough to fit in memory.

Composing a Pipeline
--------------------

In many cases, a preprocessing pipeline can be thought of as a sequence of transformations.
:class:`~pathml.preprocessing.pipeline.Pipeline` objects can be created by composing
a list of :class:`~pathml.preprocessing.transforms.Transform`:

.. code-block:: python

    pipeline = Pipeline([
        BoxBlur(kernel_size=15),
        TissueDetectionHE(mask_name = "tissue", min_region_size=500,
                          threshold=30, outer_contours_only=True)
    ])
..

In this example, the preprocessing pipeline will first apply a box blur kernel, and then apply tissue detection.
It is that easy to compose pipelines by mixing and matching :class:`~pathml.preprocessing.transforms.Transform` objects!


Custom Transforms
-----------------

A :class:`~pathml.preprocessing.pipeline.Pipeline` is a special case of
a :class:`~pathml.preprocessing.transforms.Transform` which makes it easy
to compose several :class:`~pathml.preprocessing.transforms.Transform`s sequentially.
However, in some cases, you may want to implement a :class:`~pathml.preprocessing.transforms.Transform` from scratch.
For example, you may want to apply a transformation which is not already implemented in ``PathML``.
Or, perhaps you want to apply a preprocessing pipeline which is not perfectly sequential.

To define a new custom :class:`~pathml.preprocessing.transforms.Transform`,
all you need to do is create a class which inherits from :class:`~pathml.preprocessing.transforms.Transform` and
implements an ``apply()`` method which takes a :class:`~pathml.core.tile.Tile` as an argument and modifies it in place.
You may also implement a functional method ``F()``, although that is not strictly required.

For example, let's take a look at how :class:`~pathml.preprocessing.transforms.BoxBlur` is implemented:

.. code-block:: python

    class BoxBlur(Transform):
        """Box (average) blur kernel."""
        def __init__(self, kernel_size=5):
            self.kernel_size = kernel_size

        def F(self, image):
            return cv2.boxFilter(image, ksize = (self.kernel_size, self.kernel_size), ddepth = -1)

        def apply(self, tile):
            tile.image = self.F(tile.image)
..

That's it! Once you define your custom :class:`~pathml.preprocessing.transforms.Transform`,
you can plug it in with any of the other :class:`~pathml.preprocessing.transforms.Transform`s,
compose :class:`~pathml.preprocessing.pipeline.Pipeline`, etc.
