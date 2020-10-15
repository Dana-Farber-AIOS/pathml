import openslide
import os

class Dataset:
    """
    Class representing datasets composed of slides.
    Filesystem:
        dataset
            -> paths to slides (.svs)
            -> /tiles/
                -> paths to tiles
            -> /labels/ #this is masks
                -> paths to labels (.csv)
    """
    '''
    TODO: 
        requires all tiles in each slide to be written together as one file in processedpath
            choose: 
                keep a set of paths
                write all tiles to h5
            concerns:
                dataloader design (grabbing only a few tiles might be more efficient each tile its own file)
        when generate masks (which could be used as labels), generate labels folder following filesystem
    '''
    def __init__(
            self, 
            path, 
            slidelabels=None
    ):
        self.path = path
        self.slides = {}
        self.processedpath = None
        self.tilelabelpath = None
        if slidelabels == None:
            self.slidelabels = None
        else:
            self.slidelabels = pd.read_csv(slidelabels)

        # add slide paths to self.slides
        for f in os.listdir(path):
            slidepath = os.path.join(self.path, f)
            if isopenslideformat(slidepath):
                name = os.path.splitext(f)[0]
                self.slides[name] = [slidepath, None, None] 

        # add tile paths to self.slides 
        tilepath = os.path.join(self.path, 'tiles')
        self._loadtiles(tilepath)

        # add tile labels to self.slides
        tilelabelpath = os.path.join(path, 'labels')
        self._loadtileslabels(tilelabelpath)

        def _loadtiles(self, tilepath):
            if os.path.isdir(tilepath):
                self.tilepath = tilepath
                for slidetiles in os.listdir(tilepath):
                    name = os.path.splitext(slidetiles)[0]
                    slidetilespath = os.path.join(tilespath, slidetiles)
                    self.slides[name][1] = tilepath

        def _loadtileslabels(self, tilelabelpath):
            if os.path.isdir(tilelabelpath):
                self.tilelabelpath = tilelabelpath
                for label in os.listdir(tilelabelpath):
                    name = os.path.splitext(label)[0]
                    labelpath = os.path.join(tilelabelpath, label)
                    slide.slides[name][2] = labelpath
       
def isopenslideformat(path):
    try:
        openslide.open_slide(path) 
    except:
        return False
    return True
