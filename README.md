# PathML 

Utilities for working with pathology images, facilitating machine learning for digital pathology.

## Requirements

* Install [OpenSlide](https://openslide.org/download/)  
    * Linux: ``sudo apt-get install openslide-tools``
    * Mac: ``brew install openslide``

* For managing environments, we recommend using Conda. 
    Download Miniconda [here](https://docs.conda.io/en/latest/miniconda.html)


## Installation

````
git clone https://github.com/Dana-Farber/pathml.git     # clone repo
cd pathml                               # enter repo directory
conda env create -f environment.yml     # create conda environment
conda activate pathml                   # activate conda environment
pip install -e .                        # install pathml in conda environment
````

## Generate Documentation

This repo is not yet open to the public. Once we open source it, we will host documentation online.
Until then, you must build a local copy of the documentation yourself.

````
# first install packages for generating docs
pip install ipython sphinx nbsphinx nbsphinx-link sphinx-rtd-theme  
cd docs         # enter docs directory
make html       # build docs in html format
````

Then use your favorite web browser to open ``pathml/docs/build/html/index.html``

## Getting Started

The [example notebooks](examples) are a good place start with `PathML`.

1. [Building a basic preprocessing pipeline for H&E images](examples/basic_HE.ipynb)
1. [Building a more efficient custom preprocessing pipeline](examples/advanced_HE_chunks.ipynb)
1. [Stain normalization for H&E images](examples/stain_normalization.ipynb)
1. [Nucleus detection for H&E images](examples/nucleus_detection.ipynb)

## Tests and Code Coverage 

You may optionally run the test suite to verify installation. 

To run tests:  
````
conda install pytest    # first install pytest package
python -m pytest        # run test suite
````
Note that because the testing suite tests all parts of the code base, 
this may require installing additional packages as well. 
(e.g. installation of java is required for some functionality).

You may also optionally measure code coverage, i.e. what percentage of code is covered in the testing suite.

To run tests and check code coverage:
```
c  # install coverage package for code coverage
coverage run            # run tests and calculate code coverage
coverage report         # view coverage report
coverage html           # optionally generate HTML coverage report
```

## Contributing

``PathML`` is an open source project. Consider contributing to benefit the entire community!

There are many ways to contribute to PathML, including:

* Submitting bug reports
* Submitting feature requests
* Writing documentation and examples
* Fixing bugs
* Writing code for new features
* Sharing workflows
* Sharing trained model parameters
* Sharing ``PathML`` with colleagues, students, etc.

## Contact

Questions? Comments? Suggestions? Get in touch!

[Jacob Rosenthal](mailto:Jacob_Rosenthal@dfci.harvard.edu)  
Data Science Team  
Department of Informatics and Analytics  
Dana-Farber Cancer Institute