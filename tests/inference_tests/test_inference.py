import os
import numpy as np
import onnx
import onnxruntime as ort 
import pytest 

from pathml.inference import (   
    remove_initializer_from_input,
    check_onnx_clean, 
    InferenceBase, 
    Inference, 
    HaloAIInference, 
    RemoteTestHoverNet
)


def test_remove_initializer_from_input():
    # Create a temporary ONNX model file
    model_path = "test_model.onnx"
    # temp_file = tempfile.NamedTemporaryFile(delete=False)
    # temp_file.close()

    # Create a sample ONNX model with initializer and graph input
    model = onnx.ModelProto()
    model.ir_version = 4

    # Add inputs to the graph
    input_1 = model.graph.input.add()
    input_1.name = "input_1"

    input_2 = model.graph.input.add()
    input_2.name = "input_2"

    # Add an initializer that matches one of the inputs
    initializer = model.graph.initializer.add()
    initializer.name = "input_2"

    # Save the model to a file
    onnx.save(model, model_path)

    # Call the function to remove initializers
    new_model_path = "new_model.onnx"
    remove_initializer_from_input(model_path, new_model_path)

    # Assert that the initializer has been removed from the new model
    new_model = onnx.load(new_model_path)
    input_names = [input.name for input in new_model.graph.input]
    assert initializer.name not in input_names 

    # Clean up the temporary files
    os.remove(model_path)
    os.remove(new_model_path)
    
def test_check_onnx_clean():
    # Create a temporary ONNX model file
    model_path = "test_model.onnx"
    # temp_file = tempfile.NamedTemporaryFile(delete=False)
    # temp_file.close()

    # Create a sample ONNX model with initializer and graph input
    model = onnx.ModelProto()
    model.ir_version = 4

    # Add inputs to the graph
    input_1 = model.graph.input.add()
    input_1.name = "input_1"

    input_2 = model.graph.input.add()
    input_2.name = "input_2"

    # Add an initializer that matches one of the inputs
    initializer = model.graph.initializer.add()
    initializer.name = "input_2"

    # Save the model to a file
    onnx.save(model, model_path)

    if check_onnx_clean(model_path):
        pass
    else:
        raise ValueError('check_onnx_clean function is not working') 

    # Clean up the temporary files
    os.remove(model_path)
    
def test_InferenceBase(): 
    
    # initialize InferenceBase
    test = InferenceBase()
    
    # test setter functions 
    test.set_name('name') 
    
    test.set_num_classes('num_classes')
    
    test.set_model_type('model_type')
        
    test.set_notes('notes')
    
    test.set_model_input_notes('model_input_notes')
    
    test.set_model_output_notes('model_output_notes')
    
    test.set_citation('citation')
    
    for key in test.model_card:
        assert key == test.model_card[key], f"function for {key} is not working"
    
    # test reshape function 
    random = np.random.rand(1,2,3)
    assert test.reshape(random).shape == (1, 3, 1, 2), "reshape function is not working on 3d arrays" 
    
    random = np.random.rand(1,2,3,4,5)
    assert test.reshape(random).shape == (5,4,3,2,1), "reshape function is not working on 5d arrays" 
    
def test_Inference(tileHE): 
    
    new_path = '../random_model.onnx'
    
    inference = Inference(model_path = new_path, input_name = 'data', num_classes = 1, model_type = 'segmentation')
    
    orig_im = tileHE.image
    inference.apply(tileHE)
    assert np.array_equal(tileHE.image, inference.F(orig_im))
    
def test_HaloAIInference(tileHE): 

    new_path = '../random_model.onnx'

    inference = HaloAIInference(model_path = new_path, input_name = 'data', num_classes = 1, model_type = 'segmentation')
    orig_im = tileHE.image
    inference.apply(tileHE)
    assert np.array_equal(tileHE.image, inference.F(orig_im))

def test_RemoteTestHoverNet(tileHE): 

    inference = RemoteTestHoverNet()

    orig_im = tileHE.image
    inference.apply(tileHE)
    assert np.array_equal(tileHE.image, inference.F(orig_im))
    
    inference.remove() 