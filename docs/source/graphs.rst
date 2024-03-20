Graphs
=========

``PathML`` provides a Graph API to construct cell or tissue graphs from Whole-Slide Images (WSIs).

.. note::
    Graphs are a data structure comprised of nodes connected by edges, which allow for explicit modeling of spatial relationships. 
    In computational pathology, nodes may represent tissue regions or individual nuclei, and the resulting graph structure can be 
    used to study the spatial organization of the specimen. 

We provide template code below for cell graph construction. 

.. code-block::

    # load packages
    from pathml.core import HESlide
    
    from pathml.preprocessing import Pipeline, NucleusDetectionHE
    
    from pathml.graph import KNNGraphBuilder
    from pathml.graph.utils import get_full_instance_map
    
    # Define slide path
    slide_path = 'PATH TO SLIDE'
    
    # Initialize pathml.core.slide_data.HESlide object
    wsi = HESlide(slide_path, name = slide_path, backend = "openslide", stain = 'HE')
    
    # Set up PathML pipeline for nuclei detection
    pipeline = Pipeline([NucleusDetectionHE(mask_name = "detect_nuclei")])
    
    # Run pipeline to get nuclei segmentation masks
    wsi.run(pipeline, overwrite_existing_tiles=True, distributed=False, tile_pad=True, tile_size=PATCH_SIZE)
    
    # Extract the nuclei segmentation masks
    image, nuclei_map, nuclei_centroid = get_full_instance_map(wsi, patch_size = PATCH_SIZE, mask_name="detect_nuclei")
    
    # Initialize a pathml.graph.KNNGraphBuilder object
    knn_graph_builder = KNNGraphBuilder(k=5, thresh=50, add_loc_feats=True)
    
    # Build the cell graph
    cell_graph = knn_graph_builder.process(nuclei_map, return_networkx=True)


For a full example that considers tissue graph construction and feature extraction for machine learning, please refer to the Graph construction and processing tab under Examples. 