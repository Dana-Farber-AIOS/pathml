import os

import numpy as np
import onnx
import torch

from pathml.core import SlideData
from pathml.inference import (
    HaloAIInference,
    Inference,
    InferenceBase,
    RemoteTestHoverNet,
    check_onnx_clean,
    convert_pytorch_onnx,
    remove_initializer_from_input,
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
        raise ValueError("check_onnx_clean function is not working")

    # Clean up the temporary files
    os.remove(model_path)


def test_InferenceBase():
    # initialize InferenceBase
    test = InferenceBase()

    # test setter functions
    test.set_name("name")

    test.set_num_classes("num_classes")

    test.set_model_type("model_type")

    test.set_notes("notes")

    test.set_model_input_notes("model_input_notes")

    test.set_model_output_notes("model_output_notes")

    test.set_citation("citation")

    # test model card
    for key in test.model_card:
        assert key == test.model_card[key], f"function for {key} is not working"

    # test repr function
    assert "Base class for all ONNX models" == repr(test)

    # test get model card fxn
    assert test.model_card == test.get_model_card()

    # test reshape function
    random = np.random.rand(1, 2, 3)
    assert test.reshape(random).shape == (
        1,
        3,
        1,
        2,
    ), "reshape function is not working on 3d arrays"

    random = np.random.rand(1, 2, 3, 4, 5)
    assert test.reshape(random).shape == (
        5,
        4,
        3,
        2,
        1,
    ), "reshape function is not working on 5d arrays"


def test_Inference(tileHE):
    new_path = "tests/testdata/random_model.onnx"

    inference = Inference(
        model_path=new_path, input_name="data", num_classes=1, model_type="segmentation"
    )

    orig_im = tileHE.image
    inference.apply(tileHE)
    assert np.array_equal(tileHE.image, inference.F(orig_im))

    assert repr(inference) == f"Class to handle ONNX model locally stored at {new_path}"

    # test initializer catching
    bad_model = "tests/testdata/model_with_initalizers.onnx"
    try:
        inference = Inference(
            model_path=bad_model,
            input_name="data",
            num_classes=1,
            model_type="segmentation",
        )
    except Exception as e:
        assert (
            str(e)
            == "The ONNX model still has graph initializers in the input graph. Use `remove_initializer_from_input` to remove them."
        )

    # test repr function with local set to False
    inference = Inference(
        model_path=new_path,
        input_name="data",
        num_classes=1,
        model_type="segmentation",
        local=False,
    )

    fake_model_name = "test model"
    inference.set_name(fake_model_name)

    assert (
        repr(inference)
        == f"Class to handle a {fake_model_name} from the PathML model zoo."
    )


# def test_HaloAIInference(tileHE):
#     new_path = "tests/testdata/random_model.onnx"

#     inference = HaloAIInference(
#         model_path=new_path, input_name="data", num_classes=1, model_type="segmentation"
#     )
#     orig_im = tileHE.image
#     inference.apply(tileHE)
#     assert np.array_equal(tileHE.image, inference.F(orig_im))

#     assert (
#         repr(inference)
#         == f"Class to handle HALO AI ONNX model locally stored at {new_path}"
#     )


# def test_RemoteTestHoverNet():
#     inference = RemoteTestHoverNet()

#     wsi = SlideData("tests/testdata/small_HE.svs")

#     tiles = wsi.generate_tiles(shape=(256, 256), pad=False)
#     a = 0
#     test_tile = None

#     while a == 0:
#         for tile in tiles:
#             test_tile = tile
#             a += 1

#     orig_im = test_tile.image
#     inference.apply(test_tile)
#     assert np.array_equal(test_tile.image, inference.F(orig_im))

#     assert (
#         repr(inference)
#         == "Class to handle remote TIAToolBox HoverNet test ONNX. See model card for citation."
#     )

#     inference.remove()


# def test_convert_pytorch_onnx():
#     test_tensor = torch.randn(1, 10)
#     model_test = torch.jit.load("tests/testdata/test.pt")

#     model_test.eval()

#     convert_pytorch_onnx(
#         model=model_test, dummy_tensor=test_tensor, model_name="test_export.onnx"
#     )

#     os.remove("test_export.onnx")

#     # test Value Error Statements

#     # test lines to check model input
#     try:
#         convert_pytorch_onnx(
#             model=None, dummy_tensor=test_tensor, model_name="test_export.onnx"
#         )

#     except Exception as e:
#         assert (
#             str(e)
#             == f"The model is not of type torch.nn.Module. Received {type(None)}."
#         )

#     # test lines to check model dummy input
#     try:
#         convert_pytorch_onnx(
#             model=model_test, dummy_tensor=None, model_name="test_export.onnx"
#         )

#     except Exception as e:
#         assert (
#             str(e)
#             == f"The dummy tensor needs to be a torch tensor. Received {type(None)}."
#         )
