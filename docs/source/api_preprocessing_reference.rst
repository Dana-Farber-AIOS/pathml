Preprocessing API
=================

Pipeline
--------

.. autoapiclass:: pathml.preprocessing.Pipeline

Transforms
----------

.. autoapiclass:: pathml.preprocessing.MedianBlur
.. autoapiclass:: pathml.preprocessing.GaussianBlur
.. autoapiclass:: pathml.preprocessing.BoxBlur
.. autoapiclass:: pathml.preprocessing.BinaryThreshold
.. autoapiclass:: pathml.preprocessing.MorphOpen
.. autoapiclass:: pathml.preprocessing.MorphClose
.. autoapiclass:: pathml.preprocessing.ForegroundDetection
.. autoapiclass:: pathml.preprocessing.SuperpixelInterpolation
.. autoapiclass:: pathml.preprocessing.StainNormalizationHE
.. autoapiclass:: pathml.preprocessing.NucleusDetectionHE
.. autoapiclass:: pathml.preprocessing.TissueDetectionHE
.. autoapiclass:: pathml.preprocessing.LabelArtifactTileHE
.. autoapiclass:: pathml.preprocessing.LabelWhiteSpaceHE
.. autoapiclass:: pathml.preprocessing.SegmentMIF
.. autoapiclass:: pathml.preprocessing.SegmentMIFRemote
.. autoapiclass:: pathml.preprocessing.QuantifyMIF
.. autoapiclass:: pathml.preprocessing.CollapseRunsVectra
.. autoapiclass:: pathml.preprocessing.CollapseRunsCODEX
.. autoapiclass:: pathml.preprocessing.RescaleIntensity    
.. autoapiclass:: pathml.preprocessing.HistogramEqualization
.. autoapiclass:: pathml.preprocessing.AdaptiveHistogramEqualization


TileStitching
-------------
This section covers the `TileStitcher` class, which is specialized for stitching tiled images, particularly useful in digital pathology.

.. autoapiclass:: pathml.preprocessing.tilestitcher.TileStitcher
   :members: run_image_stitching, shutdown
   :show-inheritance:
   