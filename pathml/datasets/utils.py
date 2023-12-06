"""
Copyright 2021, Dana-Farber Cancer Institute and Weill Cornell Medicine
License: GNU GPL 2.0
"""

import importlib

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from pathml.datasets.datasets import InstanceMapPatchDataset


def pannuke_multiclass_mask_to_nucleus_mask(multiclass_mask):
    """
    Convert multiclass mask from PanNuke to a single channel nucleus mask.
    Assumes each pixel is assigned to one and only one class. Sums across channels, except the last mask channel
    which indicates background pixels in PanNuke.
    Operates on a single mask.

    Args:
        multiclass_mask (torch.Tensor): Mask from PanNuke, in classification setting. (i.e. ``nucleus_type_labels=True``).
            Tensor of shape (6, 256, 256).

    Returns:
        Tensor of shape (256, 256).
    """
    # verify shape of input
    assert (
        multiclass_mask.ndim == 3 and multiclass_mask.shape[0] == 6
    ), f"Expecting a mask with dims (6, 256, 256). Got input of shape {multiclass_mask.shape}"
    assert (
        multiclass_mask.shape[1] == 256 and multiclass_mask.shape[2] == 256
    ), f"Expecting a mask with dims (6, 256, 256). Got input of shape {multiclass_mask.shape}"
    # ignore last channel
    out = np.sum(multiclass_mask[:-1, :, :], axis=0)
    return out


def _remove_modules(model, last_layer):
    """
    Remove all modules in the model that come after a given layer.

    Args:
        model (nn.Module): A PyTorch model.
        last_layer (str): Last layer to keep in the model.

    Returns:
        Model (nn.Module) without pruned modules.
    """
    modules = [n for n, _ in model.named_children()]
    modules_to_remove = modules[modules.index(last_layer) + 1 :]
    for mod in modules_to_remove:
        setattr(model, mod, nn.Sequential())
    return model


class DeepPatchFeatureExtractor:
    """
    Patch feature extracter of a given architecture and put it on GPU if available using
    Pathml.datasets.InstanceMapPatchDataset.

    Args:
        patch_size (int): Desired size of patch.
        batch_size (int): Desired size of batch.
        architecture (str or nn.Module): String of architecture. According to torchvision.models syntax, path to local model or nn.Module class directly.
        entity (str): Entity to be processed. Must be one of 'cell' or 'tissue'. Defaults to 'cell'.
        device (torch.device): Torch Device used for inference.
        fill_value (int): Value to fill outside the instance maps. Defaults to 255.
        threshold (float): Threshold for processing a patch or not.
        resize_size (int): Desired resized size to input the network. If None, no resizing is done and
            the patches of size patch_size are provided to the network. Defaults to None.
        with_instance_masking (bool): If pixels outside instance should be masked. Defaults to False.
        extraction_layer (str): Name of the network module from where the features are
            extracted.

    Returns:
        Tensor of features computed for each entity.
    """

    def __init__(
        self,
        patch_size,
        batch_size,
        architecture,
        device="cpu",
        entity="cell",
        fill_value=255,
        threshold=0.2,
        resize_size=224,
        with_instance_masking=False,
        extraction_layer=None,
    ):

        self.fill_value = fill_value
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.resize_size = resize_size
        self.threshold = threshold
        self.with_instance_masking = with_instance_masking
        self.entity = entity
        self.device = device

        if isinstance(architecture, nn.Module):
            self.model = architecture.to(self.device)
        elif architecture.endswith(".pth"):
            model = self._get_local_model(path=architecture)
            self._validate_model(model)
            self.model = self._remove_layers(model, extraction_layer)
        else:
            try:
                global torchvision
                import torchvision

                model = self._get_torchvision_model(architecture).to(self.device)
                self._validate_model(model)
                self.model = self._remove_layers(model, extraction_layer)
            except (ImportError, ModuleNotFoundError):
                raise Exception(
                    "Using online models require torchvision to be installed"
                )

        self.normalizer_mean = [0.485, 0.456, 0.406]
        self.normalizer_std = [0.229, 0.224, 0.225]

        self.num_features = self._get_num_features(patch_size)
        self.model.eval()

    @staticmethod
    def _validate_model(model):
        """Raise an error if the model does not have the required attributes."""

        if not isinstance(model, torchvision.models.resnet.ResNet):
            if not hasattr(model, "classifier"):
                raise ValueError(
                    "Please provide either a ResNet-type architecture or"
                    + ' an architecture that has the attribute "classifier".'
                )

            if not (hasattr(model, "features") or hasattr(model, "model")):
                raise ValueError(
                    "Please provide an architecture that has the attribute"
                    + ' "features" or "model".'
                )

    def _get_num_features(self, patch_size):
        """Get the number of features of a given model."""
        dummy_patch = torch.zeros(1, 3, self.resize_size, self.resize_size).to(
            self.device
        )
        features = self.model(dummy_patch)
        return features.shape[-1]

    def _get_local_model(self, path):
        """Load a model from a local path."""
        model = torch.load(path, map_location=self.device)
        return model

    def _get_torchvision_model(self, architecture):
        """Returns a torchvision model from a given architecture string."""

        module = importlib.import_module("torchvision.models")
        model_class = getattr(module, architecture)
        model = model_class(weights="IMAGENET1K_V1")
        model = model.to(self.device)
        return model

    @staticmethod
    def _remove_layers(model, extraction_layer=None):
        """Returns the model without the unused layers to get embeddings."""

        if hasattr(model, "model"):
            model = model.model
            if extraction_layer is not None:
                model = _remove_modules(model, extraction_layer)
        if isinstance(model, torchvision.models.resnet.ResNet):
            if extraction_layer is None:
                # remove classifier
                model.fc = nn.Sequential()
            else:
                # remove all layers after the extraction layer
                model = _remove_modules(model, extraction_layer)
        else:
            # remove classifier
            model.classifier = nn.Sequential()
            if extraction_layer is not None:
                # remove average pooling layer if necessary
                if hasattr(model, "avgpool"):
                    model.avgpool = nn.Sequential()
                # remove all layers in the feature extractor after the extraction layer
                model.features = _remove_modules(model.features, extraction_layer)
        return model

    @staticmethod
    def _preprocess_architecture(architecture):
        """Preprocess the architecture string to avoid characters that are not allowed as paths."""
        if architecture.endswith(".pth"):
            return f"Local({architecture.replace('/', '_')})"
        else:
            return architecture

    def _collate_patches(self, batch):
        """Patch collate function"""

        instance_indices = [item[1] for item in batch]
        patches = [item[0] for item in batch]
        patches = torch.stack(patches)
        return instance_indices, patches

    def process(self, input_image, instance_map):
        """Main processing function that takes in an input image and an instance map and returns features for all
        entities in the instance map"""

        # Create a pathml.datasets.datasets.InstanceMapPatchDataset class
        image_dataset = InstanceMapPatchDataset(
            image=input_image,
            instance_map=instance_map,
            entity=self.entity,
            patch_size=self.patch_size,
            threshold=self.threshold,
            resize_size=self.resize_size,
            fill_value=self.fill_value,
            mean=self.normalizer_mean,
            std=self.normalizer_std,
            with_instance_masking=self.with_instance_masking,
        )

        # Create a torch DataLoader
        image_loader = DataLoader(
            image_dataset,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=0,
            collate_fn=self._collate_patches,
        )

        # Initialize feature tensor
        features = torch.zeros(
            size=(len(image_dataset.entities), self.num_features),
            dtype=torch.float32,
            device=self.device,
        )
        embeddings = {}

        # Get features for batches of patches and add to feature tensor
        for instance_indices, patches in tqdm(image_loader, total=len(image_loader)):

            # Send to device
            patches = patches.to(self.device)

            # Inference mode
            with torch.no_grad():
                emb = self.model(patches).squeeze()
            for j, key in enumerate(instance_indices):

                # If entity already exists, add features on top of previous features
                if key in embeddings:
                    embeddings[key][0] += emb[j]
                    embeddings[key][1] += 1
                else:
                    embeddings[key] = [emb[j], 1]
        for k, v in embeddings.items():
            features[k, :] = v[0] / v[1]
        return features.cpu().detach()
