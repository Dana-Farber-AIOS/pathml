"""
Copyright 2021, Dana-Farber Cancer Institute and Weill Cornell Medicine
License: GNU GPL 2.0
"""

from dataclasses import dataclass


@dataclass(frozen = True)
class SlideType:
    """
    SlideType objects define types based on a set of image parameters.

    Args:
        stain (str, optional): One of ['HE', 'IHC', 'Fluor']. Flag indicating type of slide stain. Defaults to None.
        tma (bool, optional): Flag indicating whether the slide is a tissue microarray (TMA). Defaults to False.
        rgb (bool, optional): Flag indicating whether image is in RGB color. Defaults to False.
        volumetric (bool, optional): Flag indicating whether image is volumetric. Defaults to False.
        time_series (bool, optional): Flag indicating whether image is time-series. Defaults to False.

    Examples:
        >>> from pathml import SlideType, types
        >>> he_type = SlideType(stain = "HE", rgb = True)    # define slide type manually
        >>> types.HE == he_type    # can also use pre-made types for convenience
        True
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

    def asdict(self):
        return {'stain':self.stain, 'tma':self.tma, 'rgb':self.rgb, 'volumetric':self.volumetric, 'time_series':self.time_series}


class _PremadeTypes:
    # instantiations of common SlideTypes for convenience
    def __init__(self):
        self.HE = SlideType(tma = False, rgb = True, stain = 'HE', volumetric = False, time_series = False)
        self.HE_TMA = SlideType(tma = True, rgb = True, stain = 'HE', volumetric = False, time_series = False)
        self.IHC = SlideType(tma = False, rgb = True, stain = 'IHC', volumetric = False, time_series = False)
        self.IHC_TMA = SlideType(tma = True, rgb = True, stain = 'IHC', volumetric = False, time_series = False)
        self.IF = SlideType(tma = False, rgb = False, stain = 'Fluor', volumetric = False, time_series = False)
        self.IF_TMA = SlideType(tma = True, rgb = False, stain = 'Fluor', volumetric = False, time_series = False)

    def __repr__(self):
        out = "pathml.types provides pre-made slide types for convenience. Available types:\n"
        out += "'types.HE', 'types.HE_TMA', 'types.IHC', 'types.IHC_TMA', 'types.IF', 'types.IF_TMA'\n"
        out += "Please refer to documentation for pathml.core.types.SlideType"
        return out


# instantiate the class so that types are accessible in namespace, e.g. types.HE
types = _PremadeTypes()
