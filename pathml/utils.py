import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TABLEAU_COLORS

from pathml.preprocessing.utils import segmentation_lines


def plot_segmentation(ax, masks, palette=None, markersize=5):
    """
    Plot segmentation contours. Supports multi-class masks.
    
    Args:
        ax: matplotlib axis
        masks (np.ndarray): Mask array of shape (n_masks, H, W). Zeroes are background pixels.
        palette: color palette to use. if None, defaults to matplotlib.colors.TABLEAU_COLORS
        markersize (int): Size of markers used on plot. Defaults to 5
    """
    assert masks.ndim == 3
    n_channels = masks.shape[0]
    
    if palette is None:
        palette = list(TABLEAU_COLORS.values())
    
    nucleus_labels = list(np.unique(masks))
    if 0 in nucleus_labels: 
        nucleus_labels.remove(0)  # background
    # plot each individual nucleus
    for label in nucleus_labels:
        for i in range(n_channels):
            nuclei_mask = masks[i, ...] == label
            x, y = segmentation_lines(nuclei_mask.astype(np.uint8))
            ax.scatter(x, y, color = palette[i], marker = ".", s = markersize) 