import openslide
import pandas

class Dataset:
    """
    Class representing datasets composed of slides.
    """
    '''
    give path
        if the path contains slides
            create dict {slidepath, processedpath, labelpath} 
                in this case {slidepath, None, labelpath}
        if the path contains filesystem
            infer dict {slidepath, processedpath, labelpath}
                in this case {slidepath, processedpath, labelpath}
    '''
    def __init__(
            self, 
            path=None, 
            slidelabels=None
    ):
        self.path = path
        self.slides = {}
        if slidelabels=None:
            self.slidelabels = None
        else:
            self.slidelabels = pd.read_csv(slidelabels)
        # add raw slide paths to slides dict 
        for f  in os.listdir(path):
            if isopenslideformat(f):
                slidepath = f
                name = os.path.splitext(f)[0]
                self.slides[name] = [slidepath, None, None] 
        # add processed slide paths to slides dict 
        processedpath = os.path.join(path, 'processed')
        if os.path.isdir(processedpath):
            self.processedpath = processedpath
            for slide in os.listdir(processedpath):
                name = os.path.splitext(slide)[0]
                tilepath = slide
                self.slides[name][1] = tilepath
        # add patch level labels to slides dict
        tilelabels = os.path.join(path, 'labels')
        if os.path.isdir(tilelabels):
            self.tilelabels = tilelabels
            for label in os.listdir(tilelabels):
                name = os.path.splitext(label)[0]
                labelpath = label
                slide.slides[name][2] = labelpath
       
def isopenslideformat(path):
    try:
        openslide.open_slide(path) 
        return True
    except UnidentifiedImageError:
        return False
    except IsADirectoryError:
        return False
