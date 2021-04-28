"""
Copyright 2021, Dana-Farber Cancer Institute and Weill Cornell Medicine
License: GNU GPL 2.0
"""

import pickle

import pathml.core.tile
from pathml.preprocessing.transforms import Transform


class Pipeline(Transform):
    """
    Compose a sequence of Transforms

    Args:
        transform_sequence (list): sequence of transforms to be consecutively applied.
            List of `pathml.core.Transform` objects
    """
    def __init__(self, transform_sequence):
        assert all([isinstance(t, Transform) for t in transform_sequence]), f"All elements in input list must be of" \
                                                                            f" type pathml.core.Transform"
        self.transforms = transform_sequence

    def __len__(self):
        return len(self.transforms)

    def __repr__(self):
        out = f"Pipeline([\n"
        for t in self.transforms:
            out += f"\t{repr(t)},\n"
        out += "])"
        return out

    def apply(self, tile):
        # this function has side effects
        # modifies the tile in place, but also returns the modified tile
        # need to do this for dask distributed
        assert isinstance(tile, pathml.core.tile.Tile), f"argument of type {type(tile)} must be a pathml.core.Tile object."
        for t in self.transforms:
            t.apply(tile)
        return tile

    def save(self, filename):
        """
        save pipeline to disk

        Args:
            filename (str): save path on disk
        """
        pickle.dump(self, open(filename, "wb"))
