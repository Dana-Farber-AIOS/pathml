class ImageTransform:
    """
    Transforms of the form image -> image
    """
    def apply(self, image):
        """Apply transform to input image"""
        raise NotImplementedError


class SegmentationTransform:
    """
    Transforms of the form image -> mask
    """
    def apply(self, image):
        """Apply transform to input image"""
        raise NotImplementedError


class MaskTransform:
    """
    Transforms of the form mask -> mask
    """
    def apply(self, mask):
        """Apply transform to input mask"""
        raise NotImplementedError
