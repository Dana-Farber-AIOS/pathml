"""
Copyright 2021, Dana-Farber Cancer Institute and Weill Cornell Medicine
License: GNU GPL 2.0
"""

import pytest

from pathml.core import SlideType, types


@pytest.mark.parametrize("slide_type", [types.IF, types.HE, types.IHC, types.CODEX, types.Vectra])
def test_load_from_dict(slide_type):
    slide_type_dict = slide_type.asdict()
    loaded_slide_type = SlideType(**slide_type_dict)
    assert loaded_slide_type == slide_type
