import anndata
import numpy as np
import pandas as pd
from warnings import warn

try:
    import scanpy as sc
    import squidpy as sq
    import tensorly as tl
    from tensorly.decomposition import non_negative_tucker
except ImportError:
    warn(
        """The pathml.ml.multiparametric package requires extra libraries to be installed.
    You can install these via pip using:
    
    pip install scanpy squidpy tensorly
    """
    )
    raise ImportError(
        "pathml.ml.multiparametric requires scanpy squidpy and tensorly to be installed"
    ) from None


def spatialneighborhood(
        slidedata, 
        phenotypekey, 
        n_neighbors=10
    ):
    """
    Cellular neighborhood discovery by graph clustering.
    Graph based clustering by Leiden algorithm (wraps scanpy.tl.leiden).

    Requires quantified slidedata object (slidedata.count).

    Args:
        slidedata(pathml.core.SlideData): slidedata object
        phenotypekey(str): phenotype key in slidedata.counts.obs
        n_neighbors(int): number of spatial nearest neighbors 
    """
    raise NotImplementedError(f"spatialneighborhood is WIP")
    assert slidedata.counts, f"must quantify image and generate a single cell anndata object before computing neighborhood enrichment"
    assert phenotypekey in slidedata.counts.obs.keys(), f"passed key {phenotypekey} is not in slidedata.adata.obs"
    assert isinstance(n_neighbors, int), f"n_neighbors is of type {type(n_neighbors)} but must be of type int"

    # convert nearest neighbors of each cell to vector of cell types 
    if slidedata.counts.isbacked:
        adata = slidedata.counts.to_memory().copy()
    else:
        adata = slidedata.counts.copy()
    sq.gr.spatial_neighbors(adata, n_neigh=n_neighbors, key_added='spatial') 
    neighborhooddf = None 

    # create neighborhood anndata object
    # TODO: vectorize
    for cell, other in adata.obs.iterrows():
        adjacencylist = adata.obsp['spatial_connectivities']
        # TODO: why doesn't neighborids give neighbors for every cell?
        # get neighbor cells and reduce into vector
        neighborids = adjacencylist[int(adata.obs.index.get_loc(cell))]
        neighboridstuple = [str(neighbor) for neighbor in neighborids.indices.tolist()]
        neighborcells = adata[adata.obs.index.isin(neighboridstuple)]
        neighborvector = pd.DataFrame(neighborcells.obs.groupby(phenotypekey).count()['x'])
        neighborvector = neighborvector.rename(columns={'x':cell})
        neighborvectordf = pd.DataFrame(index=adata.obs.leiden.cat.categories.tolist())
        neighborvectordf = pd.concat([neighborvectordf, neighborvector], join='outer', axis=1) 
        neighborvectordf = neighborvectordf.fillna(0)
        neighborvectordf = neighborvectordf.transpose()
        if neighborhooddf is None:
            neighborhooddf = neighborvectordf
        else:
            neighborhooddf = pd.concat([neighborvectordf, neighborhooddf])
        # populate adata object with neighborhood vectors
    celladata = anndata.AnnData(X=neighborhooddf)

    # cluster neighborhood anndata object
    sc.pp.neighbors(celladata)
    sc.tl.leiden(celladata, key_added='leiden_neighborhood')

    # transfer labels
    transfer = celladata[celladata.obs.index.isin(slidedata.counts.obs.index.tolist())]
    slidedata.counts.obs['leiden_neighborhood'] = transfer.obs['leiden_neighborhood']
