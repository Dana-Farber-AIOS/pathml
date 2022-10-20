"""
Copyright 2021, Dana-Farber Cancer Institute and Weill Cornell Medicine
License: GNU GPL 2.0
"""

from .pipeline import Pipeline

# TODO: standardize transform names to be all verbs or all nouns
# e.g. NucleusDetection vs. DetectNuclei
# e.g. SegmentMIF vs. MIFSegmentation/SegmentationMIF
from .transforms import (
    BinaryThreshold,
    BoxBlur,
    ForegroundDetection,
    GaussianBlur,
    MedianBlur,
    MorphOpen,
    MorphClose,
    NucleusDetectionHE,
    HoVerNetSegmentation,
    StainNormalizationHE,
    SuperpixelInterpolation,
    TissueDetectionHE,
    LabelArtifactTileHE,
    LabelWhiteSpaceHE,
    SegmentMIF,
    QuantifyMIF,
    CollapseRunsVectra,
    CollapseRunsCODEX,
    RescaleIntensity,
    HistogramEqualization,
    AdaptiveHistogramEqualization,
)
