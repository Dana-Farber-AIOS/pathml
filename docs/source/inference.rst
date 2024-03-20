Inference
=========

``PathML`` comes with an API to run inference on ONNX models. This API relies on PathML's existing preprocessing pipeline. 

Below is an example of how to run inference on a locally stored ONNX model. 

.. code-block::
    
    # load packages
    from pathml.core import SlideData

    from pathml.preprocessing import Pipeline
    import pathml.preprocessing.transforms as Transforms

    from pathml.inference import Inference, remove_initializer_from_input

    # Define slide path
    slide_path = 'PATH TO SLIDE'

    # Set path to model 
    model_path = 'PATH TO ONNX MODEL'
    # Define path to export fixed model
    new_path = 'PATH TO SAVE NEW ONNX MODEL'

    # Fix the ONNX model by removing initializers. Save new model to `new_path`. 
    remove_initializer_from_input(model_path, new_path) 

    inference = Inference(model_path = new_path, input_name = 'data', num_classes = 8, model_type = 'segmentation')

    # Create a transformation list
    transformation_list = [
        inference
    ] 

    # Initialize pathml.core.slide_data.SlideData object
    wsi = SlideData(slide_path, stain = 'Fluor')

    # Set up PathML pipeline
    pipeline = Pipeline(transformation_list)

    # Run Inference
    wsi.run(pipeline, tile_size = 1280, level = 0)

For an end to end example and explaination of the code, please see the PathML ONNX Tutorial tab under Examples. 
