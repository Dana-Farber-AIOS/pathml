import os
import ntpath
import h5py
from pathlib import Path

import torch.utils.data
from torch.utils.data import Dataset, DataLoader

from pathml.datasets.base import BaseDataModule, BaseDataset
from pathml.ml.utils import download_url

class DFCIHCCImagingDatamodule(BaseDataModule):
    """
    DataModule for DFCIHCC imaging data. 224px image patches from prostate cancer H&E images with gleason and clinical annotation.
    """
    def __init__(self, 
            data_dir, 
            shuffle = True, 
            transforms = None,
            split = None, 
            batch_size = 32
    ):
        self.data_dir = Path(data_dir)
        self.shuffle = shuffle
        self.transforms = transforms
        self.split = split
        self.batch_size = batch_size

    def _get_dataset(self, fold_ix = None):
        return DFCIHCCImagingDataset(
            data_dir = self.data_dir,
            fold_ix = fold_ix,
            transforms = self.transforms
        )

    @property
    def train_dataloader(self):
        return data.DataLoader(
                dataset = self._get_dataset(fold_ix = 1),
                batch_size = self.batch_size,
                shuffle = self.shuffle
            )

    @property
    def valid_dataloader(self):
        return data.DataLoader(
                dataset = self._get_dataset(fold_ix = 2),
                batch_size = self.batch_size,
                shuffle = self.shuffle
            )
    
    @property
    def test_dataloader(self):
        return data.DataLoader(
                dataset = self._get_dataset(fold_ix = 3),
                batch_size = self.batch_size,
                shuffle = self.shuffle
            )

class DFCIHCCImagingDataset(BaseTileDataset):
    """
    Dataset object for DFCI/HCC data.
    Preprocesses slides into tile dataset and saves to .h5.
    Imaging data from Dana Farber Cancer Institute/Harvard Cancer Center prostate cancer database.
    Dataset consists of histology, affymetrix rna arrays, metabolon metabolomics, radiology, cnv arrays, mutation annotations from gelb.
    """
    def __init__(self,
            data_dir,
            fold_ix = None,
            transforms = None
    ):
        self.data_dir = Path(data_dir)
        assert fold_ix in [1,2,3,None], f"Error: split {split} must be in [1,2,3,None]."
        self.fold_ix = fold_ix
        # if no .h5 file, generate from raw
        if not any(fname.endswith('.h5') for fname in os.listdir(self.data_dir)):
            slides = []
            for root, dirs, files in os.walk(self.data_dir):
                for file in files:
                    if file.endswith(".svs"):
                        slides.append(file)
            self.slides = slides
            self._h5fromraw()
        self.data = h5py.File(self.data_dir / "dfcihccdataset.h5", "r") 
        # TODO: grab data based on fold_ix 

    def __len__(self):
        return len(self.data['tiles'])

    def __getitem__(self, idx):
        slide = HESlide(path=self.slides[idx]) 
        head, tail = ntpath.split(self.slides[idx])
        name = os.path.splittext(tail)[0]
        return slide, name

    def _h5fromraw():
        """
        Process wsi to h5py files.
        Saves tiles to disk as side-effect.
        """
        # generate tiles from raw .svs
        pipeline = Pipeline(
                slide_loader = DFCIHCCSlideLoader(level=2),
                slide_preprocessor = DFCIHCCSlidePreprocessor(),
                tile_extractor = DFCIHCCTileExtractor(tile_size=244, chunk_size_low_res=224*4),
                tile_preprocessor = DFCIHCCTilePreprocessor()
        )
        Pipeline.run(self)

        # create h5 
        with h5py.File("dfcihccdataset.h5", "w") as f:
            # TODO: more efficient tile dtype?
            tiles = f.create_dataset("tiles", (224,), dtype='f')
            names = f.create_dataset("names", dtype=str)
            # add tiles and names 
            i = 1
            for root, dirs, files in os.walk(root):
                for file in files:
                    if file.endswith(".png"):
                        tile = cv2.imread(file)
                        head, tail = ntpath.split(file)
                        name = os.path.splittext(tail)[0]
                        tiles[i] = tile
                        names[i] = name
                        i = i+1
                # TODO: how to store fold information? in file name? in .h5 as label?
                # https://groups.google.com/g/h5py/c/P-fJCLGxcuc
            f.close()

    class DFCIHCCSlideLoader(BaseSlideLoader):
        def __init__(self, level):
            self.level = level
        
        def apply(self, path):
            data = HESlide(path).load_data(level=self.level)
            return data

    class DFCIHCCSlidePreprocessor(BaseSlidePreprocessor):
        def apply(self, data):
            tissue_detector = TissueDetectionHE(
                foreground_detection = ForegroundDetection(min_region_size=1000, max_hole_size=1000)
            )
            tissue_mask = tissue_detector.apply(data.image)
            data.mask = tissue_mask
            return data

    class DFCIHCCTileExtractor(BaseTileExtractor):
        def __init__(self, tile_size=224, chunk_size_low_res = 1000):
            self.tile_size = 224
            # size of each chunk, at low-resolution
            self.chunk_size_low_res = chunk_size_low_res

        def apply(self, data):
            """
            Use the downsampled data.mask to get full-resolution tiles.
            Process full-resolution image in chunks.
            """
            # get scale for upscaling mask to full-res
            scale = data.wsi.slide.level_downsamples[data.level]
            scale = int(scale)
            # size of each chunk, at low-resolution
            chunk_size_low_res = self.chunk_size_low_res
            # size of each chunk, at full-resolution
            chunk_size = chunk_size_low_res * scale
            # how many chunks in each full_res dim
            # note that openslide uses (width, height) format
            full_res_j, full_res_i = data.wsi.slide.level_dimensions[0]
            # loop thru chunks
            n_chunk_i = full_res_i // chunk_size
            n_chunk_j = full_res_j // chunk_size

            for ix_i in range(n_chunk_i):
                for ix_j in range(n_chunk_j):
                    # get mask
                    mask = data.mask[ix_i*chunk_size_low_res:(ix_i + 1)*chunk_size_low_res,
                                     ix_j*chunk_size_low_res:(ix_j + 1)*chunk_size_low_res]

                    if mask.mean() == 0.0:
                        # empty chunk, no need to continue processing
                        continue
                    # upscale mask to match full-res image
                    mask_upsampled = upsample_array(mask, scale)
                    # get full-res image
                    region = data.wsi.slide.read_region(
                        location = (ix_j*chunk_size, ix_i*chunk_size),
                        level = 0, size = (chunk_size, chunk_size)
                    )
                    region_rgb = pil_to_rgb(region)

                    # divide into tiles
                    good_tiles = extract_tiles_with_mask(
                        im = region_rgb,
                        tile_size = self.tile_size,
                        mask = mask_upsampled
                    )

                    for tile in good_tiles:
                        # adjust i and j coordinates for each tile to account for the chunk offset
                        tile.i += ix_i*chunk_size
                        tile.j += ix_j*chunk_size

                    # add extracted tiles to data.tiles
                    data.tiles = good_tiles if data.tiles is None else data.tiles + good_tiles
            return data

    class DFCIHCCTilePreprocessor(BaseTilePreprocessor):
        """
        Simple tile preprocessor which applies color normalizations,
        filters out whitespace tiles, and writes tiles to disk
        """
        def apply(self, data):
            normalizer = StainNormalizationHE(stain_estimation_method='macenko')
            # save the processed tiles to a new directory in same location as original wsi
            out_dir = os.path.join(
                os.path.dirname(data.wsi.path),
                os.path.splitext(os.path.basename(data.wsi.path))[0] + "_tiled"
            )
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            # extra step to filter out whitespace tiles
            data.tiles[:] = [tile for tile in data.tiles if not label_whitespace_HE(tile.array)]
            # now loop through tiles, normalize the color, and save to disk
            for tile in data.tiles:
                tile.array = normalizer.apply(tile.array)
                tile.save(out_dir = out_dir, filename = f"{data.wsi.name}_{tile.i}_{tile.j}.png")
            return data
