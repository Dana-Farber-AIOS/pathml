"""
Copyright 2023, Dana-Farber Cancer Institute and Weill Cornell Medicine
License: GNU GPL 2.0
"""

import os
import numpy as np
import pathml.preprocessing.transforms as Transforms
import onnx
import onnxruntime
import requests


def remove_initializer_from_input(model_path, new_path):
    """Removes initializers from HaloAI ONNX models
    Taken from https://github.com/microsoft/onnxruntime/blob/main/tools/python/remove_initializer_from_input.py

    Args:
        model_path (str): path to ONNX model,
        new_path (str): path to save adjusted model w/o initializers,

    Returns:
        ONNX model w/o initializers to run inference using PathML
    """

    model = onnx.load(model_path)

    inputs = model.graph.input
    name_to_input = {}
    for onnx_input in inputs:
        name_to_input[onnx_input.name] = onnx_input

    for initializer in model.graph.initializer:
        if initializer.name in name_to_input:
            inputs.remove(name_to_input[initializer.name])

    onnx.save(model, new_path)


def check_onnx_clean(model_path):
    """Checks if the model has had it's initalizers removed from input graph.
    Adapted from from https://github.com/microsoft/onnxruntime/blob/main/tools/python/remove_initializer_from_input.py

    Args:
        model_path (str): path to ONNX model,

    Returns:
        Boolean if there are initializers in input graph.
    """

    model = onnx.load(model_path)

    inputs = model.graph.input
    name_to_input = {}
    for onnx_input in inputs:
        name_to_input[onnx_input.name] = onnx_input

    for initializer in model.graph.initializer:
        if initializer.name in name_to_input:
            return True


# Base class
# I think this should still inherit from Transforms to make the tiling easier/so we don't have to rewrite so much existing code
class InferenceBase(Transforms.Transform):
    """
    Base class for all ONNX Models.
    Each transform must operate on a Tile.
    """

    def __init__(self):
        self.model_card = {
            "name": None,
            "num_classes": None,
            "model_type": None,
            "notes": None,
            "model_input_notes": None,
            "model_output_notes": None,
            "citation": None,
        }

    def __repr__(self):
        return "Base class for all ONNX models"

    def get_model_card(self):
        return self.model_card

    def set_name(self, name):
        self.model_card["name"] = name

    def set_num_classes(self, num):
        self.model_card["num_classes"] = num

    def set_model_type(self, model_type):
        self.model_card["model_type"] = model_type

    def set_notes(self, note):
        self.model_card["notes"] = note

    def set_model_input_notes(self, note):
        self.model_card["model_input_notes"] = note

    def set_model_output_notes(self, note):
        self.model_card["model_output_notes"] = note

    def set_citation(self, citation):
        self.model_card["citation"] = citation

    def reshape(self, image):
        """standard reshaping of tile image"""
        # flip dimensions
        # follows convention used here https://github.com/Dana-Farber-AIOS/pathml/blob/master/pathml/ml/dataset.py

        if image.ndim == 3:
            # swap axes from HWC to CHW
            image = image.transpose(2, 0, 1)
            # add a dimesion bc onnx models usually have batch size as first dim: e.g. (1, channel, height, width)
            image = np.expand_dims(image, axis=0)

            return image
        else:
            # in this case, we assume that we have XYZCT channel order
            # so we swap axes to TCZYX for batching
            # note we are not adding a dim here for batch bc we assume that subsetting will create a batch "placeholder" dim
            image = image.T

            return image

    def F(self, target):
        """functional implementation"""
        raise NotImplementedError

    def apply(self, tile):
        """modify Tile object in-place"""
        raise NotImplementedError


# class to handle local onnx models
class Inference(InferenceBase):
    """Transformation to run inferrence on ONNX model.

    Assumptions:
        - The ONNX model has been cleaned by `remove_initializer_from_input` first

    Args:
        model_path (str): path to ONNX model w/o initializers,
        input_name (str): name of the input the ONNX model accepts
    """

    def __init__(
        self,
        model_path=None,
        input_name="data",
        num_classes=None,
        model_type=None,
        local=True,
    ):
        super().__init__()

        self.input_name = input_name
        self.num_classes = num_classes
        self.model_type = model_type
        self.local = local

        if self.local:
            # using a local onnx model
            self.model_path = model_path
        else:
            # if using a model from the model zoo, set the local path to a temp file
            self.model_path = "temp.onnx"

        # fill in parts of the model_card with the following info
        self.model_card["num_classes"] = self.num_classes
        self.model_card["model_type"] = self.model_type

        # check if there are initializers in input graph if using a local model
        if local:
            if check_onnx_clean(model_path):
                raise ValueError(
                    "The ONNX model still has graph initializers in the input graph. Use `remove_initializer_from_input` to remove them."
                )
        else:
            pass

    def __repr__(self):
        if self.local:
            return f"Class to handle ONNX model locally stored at {self.model_path}"
        else:
            return f"Class to handle a {self.model_card['model_name']} from the PathML model zoo."

    def inference(self, image):
        # reshape the image
        image = self.reshape(image)

        # load fixed model
        onnx_model = onnx.load(self.model_path)

        # check tile dimensions match ONNX input dimensions
        input_node = onnx_model.graph.input

        dimensions = []
        for input in input_node:
            if input.name == self.input_name:
                input_shape = input.type.tensor_type.shape.dim
                for dim in input_shape:
                    dimensions.append(dim.dim_value)

        assert (
            image.shape[-1] == dimensions[-1] and image.shape[-2] == dimensions[-2]
        ), f"expecting tile shape of {dimensions[-2]} by {dimensions[-1]}, got {image.shape[-2]} by {image.shape[-1]}"

        # check onnx model
        onnx.checker.check_model(onnx_model)

        # start an inference session
        ort_sess = onnxruntime.InferenceSession(self.model_path)

        # create model output, returns a list
        model_output = ort_sess.run(None, {self.input_name: image.astype("f")})

        return model_output

    def F(self, image):
        # run inference function
        prediction_map = self.inference(image)

        # single task model
        if len(prediction_map) == 1:
            # return first and only prediction array in the list
            return prediction_map[0]

        # multi task model
        else:
            # concatenate prediction results
            # assumes that the tasks all output prediction arrays of same dimension on H and W
            # To Do: figure out solution for way different tasks such as if a model does both segmentation and classification
            result_array = np.concatenate(prediction_map, axis=1)
            return result_array

    def apply(self, tile):
        tile.image = self.F(tile.image)


class HaloAIInference(Inference):
    """Transformation to run inferrence on HALO AI ONNX model.

    Assumptions:
        - Assumes that the ONNX model returns a tensor in which there is one prediction map for each class
        - For example, if there are 5 classes, the ONNX model will output a (1, 5, Height, Weight) tensor
        - If you select to argmax the classes, the class assumes a softmax or sigmoid has already been applied
        - HaloAI ONNX models always have 20 class maps so you need to index into the first x maps if you have x classes


    Args:
        model_path (str): path to ONNX model w/o initializers,
        num_classes (int): number of classes in the data,
        input_name (str): name of the input the ONNX model accepts
    """

    def __init__(
        self,
        model_path=None,
        input_name="data",
        num_classes=None,
        model_type=None,
        local=True,
    ):
        super().__init__(model_path, input_name, num_classes, model_type, local)

        self.model_card["num_classes"] = self.num_classes
        self.model_card["model_type"] = self.model_type

    def __repr__(self):
        return f"Class to handle HALO AI ONNX model locally stored at {self.model_path}"

    def F(self, image):
        prediction_map = self.inference(image)

        prediction_map = prediction_map[0][:, 0 : self.num_classes, :, :]

        return prediction_map

    def apply(self, tile):
        tile.image = self.F(tile.image)


# class to handle remote onnx models
# ToDo create function to remove model after tiling is done would be a sep line in workflow
class RemoteTestHoverNet(Inference):
    """Transformation to run inferrence on ONNX model.

    Citation for model:
    Pocock J, Graham S, Vu QD, Jahanifar M, Deshpande S, Hadjigeorghiou G, Shephard A, Bashir RM, Bilal M, Lu W, Epstein D.
    TIAToolbox as an end-to-end library for advanced tissue image analytics. Communications medicine. 2022 Sep 24;2(1):120.

    Args:
        model_path (str): temp file name to download onnx from huggingface,
        input_name (str): name of the input the ONNX model accepts
    """

    def __init__(
        self,
        model_path="temp.onnx",
        input_name="data",
        num_classes=5,
        model_type="Segmentation",
        local=False,
    ):
        super().__init__(model_path, input_name, num_classes, model_type, local)

        # specify URL of the model in PathML public repository
        url = "https://huggingface.co/pathml/test/resolve/main/hovernet_fast_tiatoolbox_fixed.onnx"

        # download model, save as temp.onnx
        with open(self.model_path, "wb") as out_file:
            content = requests.get(url, stream=True).content
            out_file.write(content)

        self.model_card["num_classes"] = self.num_classes
        self.model_card["model_type"] = self.model_type
        self.model_card["name"] = "Tiabox HoverNet Test"
        self.model_card["model_input_notes"] = "Accepts tiles of 256 x 256"
        self.model_card[
            "citation"
        ] = "Pocock J, Graham S, Vu QD, Jahanifar M, Deshpande S, Hadjigeorghiou G, Shephard A, Bashir RM, Bilal M, Lu W, Epstein D. TIAToolbox as an end-to-end library for advanced tissue image analytics. Communications medicine. 2022 Sep 24;2(1):120."

    def __repr__(self):
        return "Class to handle remote TIAToolBox HoverNet test ONNX. See model card for citation."

    def apply(self, tile):
        tile.image = self.F(tile.image)

    def remove(self):
        # remove the temp.onnx model
        os.remove(self.model_path)