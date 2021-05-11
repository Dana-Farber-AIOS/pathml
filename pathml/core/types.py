"""
Copyright 2021, Dana-Farber Cancer Institute and Weill Cornell Medicine
License: GNU GPL 2.0
"""

from dataclasses import dataclass


@dataclass(frozen = True)
class SlideType:
    """
    SlideType objects define types based on a set of image parameters.

    We also provide instantiations of common slide types for convenience:

    ==============  =======  ======= =======  ==========  ===========
    Type            stain    rgb     tma      volumetric  time_series
    ==============  =======  ======= =======  ==========  ===========
    ``types.HE``    'HE'     True    False    False       False
    ``types.IHC``   'IHC'    True    False    False       False
    ``types.IF``    'Fluor'  False   False    False       False
    ==============  =======  ======= =======  ==========  ===========

    Attributes:
        stain (str, optional): One of ['HE', 'IHC', 'Fluor']. Flag indicating type of slide stain. Defaults to None.
        tma (bool, optional): Flag indicating whether the slide is a tissue microarray (TMA). Defaults to False.
        rgb (bool, optional): Flag indicating whether image is in RGB color. Defaults to False.
        volumetric (bool, optional): Flag indicating whether image is volumetric. Defaults to False.
        time_series (bool, optional): Flag indicating whether image is time-series. Defaults to False.
    """
    stain: str = None
    tma: bool = False
    rgb: bool = False
    volumetric: bool = False
    time_series: bool = False

    def __post_init__(self):
        """validate the inputs here"""
        valid_stains = {'HE', 'IHC', 'Fluor'}
        if self.stain and self.stain not in valid_stains:
            raise ValueError(f"Input stain {self.stain} not valid. Must be one of {valid_stains}")


# instantiations of common slide_types for convenience
HE = SlideType(tma = False, rgb = True, stain = 'HE', volumetric = False, time_series = False)
HE_TMA = SlideType(tma = True, rgb = True, stain = 'HE', volumetric = False, time_series = False)
IHC = SlideType(tma = False, rgb = True, stain = 'IHC', volumetric = False, time_series = False)
IHC_TMA = SlideType(tma = True, rgb = True, stain = 'IHC', volumetric = False, time_series = False)
IF = SlideType(tma = False, rgb = False, stain = 'Fluor', volumetric = False, time_series = False)
