from pathml.preprocessing.base_preprocessor import (BaseSlideLoader, BaseSlidePreprocessor,
                                                    BaseTileExtractor, BaseTilePreprocessor)
import os

class Pipeline:
    """
    Object for running preprocessing pipelines.

    :param slide_loader: preprocessor which loads slide from disk
    :type slide_loader: :class:`~pathml.preprocessing.base_preprocessor.BaseSlideLoader`
    :param slide_preprocessor: preprocessor to apply on slide level
    :type slide_preprocessor: :class:`~pathml.preprocessing.base_preprocessor.BaseSlidePreprocessor`
    :param tile_extractor: preprocessor to extract tiles
    :type tile_extractor: :class:`~pathml.preprocessing.base_preprocessor.BaseTileExtractor`
    :param tile_preprocessor: preprocessor to run on each tile
    :type tile_preprocessor: :class:`~pathml.preprocessing.base_preprocessor.BaseTilePreprocessor`
    """

    def __init__(self, slide_loader, slide_preprocessor, tile_extractor, tile_preprocessor):
        assert isinstance(slide_loader, BaseSlideLoader), \
            f"slide_loader is of type {type(slide_loader)}. Must inherit from BaseSlideLoader"
        assert isinstance(slide_preprocessor, BaseSlidePreprocessor), \
            f"slide_preprocessor is of type {type(slide_preprocessor)}. Must inherit from BaseSlidePreprocessor"
        assert isinstance(tile_extractor, BaseTileExtractor), \
            f"tile_extractor is of type {type(tile_extractor)}. Must inherit from BaseTileExtractor"
        assert isinstance(tile_preprocessor, BaseTilePreprocessor), \
            f"tile_preprocessor is of type {type(tile_preprocessor)}. Must inherit from BaseTilePreprocessor"

        self.slide_loader = slide_loader
        self.slide_preprocessor = slide_preprocessor
        self.tile_extractor = tile_extractor
        self.tile_preprocessor = tile_preprocessor

    def load_slide(self, path):
        """Run only the slide_loader component of the pipeline"""
        data = self.slide_loader.apply(path)
        return data

    def run_slide_level(self, data):
        """Run only the slide_preprocessor component of the pipeline"""
        data = self.slide_preprocessor.apply(data)
        return data

    def extract_tiles(self, data):
        """Run only the tile_extractor component of the pipeline"""
        data = self.tile_extractor.apply(data)
        return data

    def run_tile_level(self, data, dataset=None):
        """Run only the tile_preprocessor component of the pipeline"""
        data = self.tile_preprocessor.apply(data, dataset)
        return data

    def run(self, path=None, dataset=None):
        """
        Run full preprocessing pipeline

        :param path: path to input WSI
        :type path: str
        :param dataset: input Dataset object
        :type dataset: :class:`~pathml.datasets.dataset.Dataset`
        :return: :class:`~pathml.preprocessing.slide_data.SlideData` object resulting from running full pipeline on
            input image
        :return: :class:`~pathml.datasets.dataset.Dataset' object containing references to processed data generated 
            by pipeline
        """
        if path:
            data = self.load_slide(path)
            data = self.run_slide_level(data)
            data = self.extract_tiles(data)
            data = self.run_tile_level(data)
            return data
        
        elif dataset:
            slides = dataset.slides
            if not dataset.tilepath:
                tilepath = os.path.join(dataset.path,'tiles')
                dataset.tilepath = tilepath
                os.makedirs(tilepath)
            for slide in slides:
                name = slide
                slidepath = slides[slide][0]
                data = self.load_slide(slidepath)
                data = self.run_slide_level(data)
                data = self.extract_tiles(data)
                data = self.run_tile_level(data,dataset)
            return dataset

        elif None:
            print('provide path or dataset')
                   
'''
# TODO:  write tiles to h5 instead of individually to disk
def write_tiles(tiles, name, path):
    """ 
    write an array of tiles to h5

    :param tiles: tiles object
    :type tiles: :class:`~pathml.preprocessing.Tile`
    :param name: slide name
    :type name: str
    :param path: save path
    :type path: str
    """

    # init new hdf5 file
    file = h5py.File(os.path.join(path,f"{name}.h5"), "w")

    # write tiles to hdf5
    dataset = file.create_dataset(
        "tiles", np.shape(tiles), h5py.h5t.STD_U8BE, data=tiles
    )
    file.close()
'''
