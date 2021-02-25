import zipfile
import os
import shutil
import sys
import numpy as np
import torch
import torch.utils.data as data
from warnings import warn
from pathlib import Path
import cv2

from pathml.datasets.base import BaseSlideDataset, BaseDataModule
from pathml.datasets.utils import download_from_url
from pathml.preprocessing.transforms import TissueDetectionHE
from pathml.preprocessing.pipeline import Pipeline
from pathml.core.slide_classes import HESlide
from pathml.core.masks import Masks

import cProfile
import pstats

class PesoDataModule(BaseDataModule):
    def __init__(self,
            data_dir,
            download=False,
            shuffle=True,
            transforms=None,
            split=None,
            batch_size=8):
        self.data_dir = Path(data_dir)
        self.transforms = transforms
        self.download = download
        if download:
            self._download_peso(self.data_dir)

    def _get_dataset(self, fold_ix=None):
        return PesoDataset(
                data_dir = self.data_dir,
                fold_ix = fold_ix,
                transforms = self.transforms
        )

    def __repr__(self):
        return f"repr=(DataModule for PESO segmentation dataset)"

    def _download_peso(self, download_dir):
        # TODO: check hash
        if not os.path.isdir(download_dir):
            print("Downloading Peso Dataset. Total file size is ~100GB, please wait.")
            files = ['peso_testset_mapping.csv','peso_testset_png.zip','peso_testset_png_padded.zip','peso_testset_regions.zip','peso_testset_wsi_1.zip','peso_testset_wsi_2.zip','peso_testset_wsi_3.zip','peso_testset_wsi_4.zip','peso_training_colordeconvolution.zip','peso_training_masks.zip','peso_training_masks_corrected.zip','peso_training_wsi_1.zip','peso_training_wsi_2.zip','peso_training_wsi_3.zip','peso_training_wsi_4.zip','peso_training_wsi_5.zip','peso_training_wsi_6.zip']
            url = f'https://zenodo.org/record/1485967/files/'
            for file in files:
                print(f"downloading {file}")
                download_from_url(f"{url}{file}", f"{download_dir}") 
            for root, _, files in os.walk(download_dir): 
                for file in files:
                    print(f"unzipping {file}")
                    if zipfile.is_zipfile(f"{root}/{file}"):
                        with zipfile.ZipFile(f"{root}/{file}",'r') as zip_ref:
                            zip_ref.extractall(f"{root}/{Path(file).stem}")
                        os.remove(f"{root}/{file}")
        trainingwsifolders = [
                'peso_training_wsi_1',
                'peso_training_wsi_2',
                'peso_training_wsi_3',
                'peso_training_wsi_4',
                'peso_training_wsi_5',
                'peso_training_wsi_6'
        ]
        for trainingwsifolder in trainingwsifolders:
            for file in os.listdir(Path(download_dir)/Path(trainingwsifolder)):
                if file.endswith('.tif'):
                    
                    profile = cProfile.Profile()
                    profile.enable()

                    name = '_'.join(file.split('_')[:-1])
                    maskpath = Path(name+'_HE_training_mask.tif') 
                    mask = HESlide(filepath = str(Path(download_dir)/Path('peso_training_masks')/maskpath), name = name)
                    shape1, shape2 = mask.slide.get_image_shape()
                    shape = (shape2, shape1)
                    print(f"image shape is {shape}")
                    mask = mask.slide.slide.read_region(((0,0)), 0, shape)
                    mask, _, _, _ = mask.split()
                    mask = np.array(mask)
                    # mask.point(lambda x: x * 255) # optionally convert to dynamic range 255 for img
                    masks = {'stroma': mask}
                    masks = Masks(masks)
                    wsi = HESlide(str(Path(download_dir)/Path(trainingwsifolder)/Path(file)) , masks = masks)
                    pipeline = Pipeline([
                        TissueDetectionHE(mask_name = 'tissue', min_region_size = 500,
                            threshold = 30, outer_contours_only = True)
                        ])
                    # TODO: choose tile size
                    wsi.run(pipeline, tile_size=250)
                    profile.disable()
                    ps = pstats.Stats(profile)
                    ps.print_stats()
                    wsi.write(str(Path(download_dir)/Path('h5')/Path(name+'.h5')))
                    os.remove(str(Path(download_dir)/Path(traininwsifolder)/Path(file)))
                    os.remove(str(Path(download_dir)/Path('peso_training_masks')/maskpath))

        else:
            warn(f'download_dir exists, download canceled')


    @property
    def train_dataloader(self):
        """
        Dataloader for training set.
        """
        return data.DataLoader(
            dataset = self._get_dataset(fold_ix = self.split),
            batch_size = self.batch_size,
            shuffle = self.shuffle
        )

    @property
    def valid_dataloader(self):
        """
        Dataloader for validation set.
        """
        if self.split in [1, 3]:
            fold_ix = 2
        else:
            fold_ix = 1
        return data.DataLoader(
            self._get_dataset(fold_ix = fold_ix),
            batch_size = self.batch_size,
            shuffle = self.shuffle
        )

    @property
    def test_dataloader(self):
        """
        Dataloader for test set.
            """
        if self.split in [1, 2]:
            fold_ix = 3
        else:
            fold_ix = 1
        return data.DataLoader(
            self._get_dataset(fold_ix = fold_ix),
            batch_size = self.batch_size,
            shuffle = self.shuffle
        )

class PesoDataset(BaseSlideDataset):
    """
    Dataset object for Peso dataset.
    Raw data downloads:
    IHC color deconvolution (n=62) with p63, ck8/18 stainings
    training masks (n=62) generated by segmenting IHC with UNet 
    training masks corrected (n=25) with manual annotations
    wsis (n=62) wsis at 0.48 \mu/pix

    testing data:
    testset regions collection of xml files with outlines of test regions
    testset png 2500x2500 pixel test regions
    testset png padded 3500x3500 pixel regions 500pixel padding
    testset mapping csv file maps test set (1-160) to xml files, benign/cancer labels
    testset wsi (n=40) at 0.48 \mu/pix

    Each WSI and associated mask is loaded as SlideData and saved in h5path format
    """

    def __init__(self,
            data_dir,
            fold_ix = None,
            transforms = None):
        self.fold_ix = fold_ix
        self.transforms = transforms

        self.data_dir = Path(data_dir)

        assert data_dir.isdir(), f"Error: data not found at {data_dir}"

        if not any(fname.endswith('.h5') for fname in os.listdir(self.data_dir / Path('h5'))):
            raise Exception('must download dataset from pathml.datasets') 
         
        getitemdict = {}
        items = 0
        for h5 in os.listdir(self.data_dir / Path('h5')):
            wsi = read(self.data_dir / Path('h5') / Path(file))
            for i in range(len(wsi.tile)):
                getitemdict[items] = (wsi.name, i) 
                items = items + 1
        self.getitemdict = getitemdict
        self.wsi = None

    def __len__(self):
        return len(self.getitemdict)

    def __getitem__(self, ix):
        wsiname, index = self.getitemdict[ix]
        if self.wsi is None or self.wsi.name != wsiname:
            self.wsi = read(self.data_dir / Path('h5') / Path(wsiname + '.h5'))
        tile = self.wsi.tiles[index] 
        return tile
