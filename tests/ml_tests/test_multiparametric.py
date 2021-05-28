"""
Copyright 2021, Dana-Farber Cancer Institute and Weill Cornell Medicine
License: GNU GPL 2.0
"""

import pytest
import scanpy as sc

from pathml.core import VectraSlide, Tiles
from pathml.ml.multiparametric import spatialneighborhood

def test_spatialneighborhood(tileVectra, anndata):
    tiles = [tileVectra]
    adata = anndata
    slidedata = VectraSlide("tests/testdata/small_vectra.qptiff", name='vectraslide', tiles=tiles)
    slidedata.counts = adata
    with pytest.raises((AssertionError, ValueError)):
        spatialneighborhood(slidedata, phenotypekey='leiden')
    sc.pp.neighbors(slidedata.counts)
    sc.tl.leiden(slidedata.counts)
    sc.tl.pca(slidedata.counts)
    slidedata.counts.obsm['spatial'] = slidedata.counts.obsm['X_pca']
    print(slidedata.counts.obsm)
    spatialneighborhood(slidedata, phenotypekey='leiden', n_neighbors = 2)
    assert slidedata.counts.obs.leiden_neighborhood
