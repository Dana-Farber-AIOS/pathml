"""
Copyright 2021, Dana-Farber Cancer Institute and Weill Cornell Medicine
License: GNU GPL 2.0
"""
from loguru import logger

class SlideType:
    """
    SlideType objects define types based on a set of image parameters.

    Args:
        stain (str, optional): One of ['HE', 'IHC', 'Fluor']. Flag indicating type of slide stain. Defaults to None.
        platform (str, optional): Flag indicating the imaging platform (e.g. CODEX, Vectra, etc.).
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

    def __init__(
        self,
        stain=None,
        platform=None,
        tma=None,
        rgb=None,
        volumetric=None,
        time_series=None,
    ):
        # verify inputs
        valid_stains = {"HE", "IHC", "Fluor"}
        if stain and stain not in valid_stains:
            raise ValueError(
                logger.exception(
                    f"Input stain {stain} not valid. Must be one of {valid_stains}"
                )
            )
        valid_platforms = {"CODEX", "Vectra", "Visium"}
        if platform and platform not in valid_platforms:
            raise ValueError(
                logger.exception(
                    f"Input stain {platform} not valid. Must be one of {valid_platforms}"
                )
            )

        # h5py can't write None as attributes, so we need to save them as zeros or empty str and interpret as None
        if stain == "":
            stain = None
        if platform == "":
            platform = None
        if tma == 0:
            tma = None
        if rgb == 0:
            rgb = None
        if volumetric == 0:
            volumetric = None
        if time_series == 0:
            time_series = None

        self.stain = stain
        self.platform = platform
        self.tma = tma
        self.rgb = rgb
        self.volumetric = volumetric
        self.time_series = time_series

    def asdict(self):
        """
        Convert to a dictionary.
        None values are represented as zeros and empty strings for compatibility
        with h5py attributes.

        If ``a`` is a SlideType object, then ``a == SlideType(**a.asdict())`` will be ``True``.
        """
        return {
            "stain": self.stain if self.stain else "",
            "platform": self.platform if self.platform else "",
            "tma": self.tma if self.tma is not None else 0,
            "rgb": self.rgb if self.rgb is not None else 0,
            "volumetric": self.volumetric if self.volumetric is not None else 0,
            "time_series": self.time_series if self.time_series is not None else 0,
        }

    def __repr__(self):
        out = f"SlideType(stain={self.stain}, platform={self.platform}, tma={self.tma}, rgb={self.rgb}, "
        out += f"volumetric={self.volumetric}, time_series={self.time_series})"
        return out

    def __eq__(self, other):
        return all(
            [
                self.stain == other.stain,
                self.platform == other.platform,
                self.tma == other.tma,
                self.rgb == other.rgb,
                self.volumetric == other.volumetric,
                self.time_series == other.time_series,
            ]
        )


class _PremadeTypes:
    # instantiations of common SlideTypes for convenience
    def __init__(self):
        self.HE = SlideType(
            stain="HE",
            platform=None,
            tma=False,
            rgb=True,
            volumetric=False,
            time_series=False,
        )
        self.IHC = SlideType(
            stain="IHC",
            platform=None,
            tma=False,
            rgb=True,
            volumetric=False,
            time_series=False,
        )
        self.IF = SlideType(
            stain="Fluor",
            platform=None,
            tma=False,
            rgb=False,
            volumetric=False,
            time_series=False,
        )
        self.CODEX = SlideType(
            stain="Fluor",
            platform="CODEX",
            tma=False,
            rgb=False,
            volumetric=False,
            time_series=False,
        )
        self.Vectra = SlideType(
            stain="Fluor",
            platform="Vectra",
            tma=False,
            rgb=False,
            volumetric=False,
            time_series=False,
        )

    def __repr__(self):
        out = "pathml.types provides pre-made slide types for convenience. Available types include:\n"
        out += "'types.HE', 'types.IHC', 'types.IF', 'types.CODEX', 'types.Vectra'\n"
        out += "Please refer to documentation for pathml.core.SlideType"
        return out


# instantiate the class so that types are accessible in namespace, e.g. types.HE
types = _PremadeTypes()
