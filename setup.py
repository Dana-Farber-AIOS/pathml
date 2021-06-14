import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pathml",
    version="1.0.0",
    author="Jacob Rosenthal et al.",
    author_email="PathML@dfci.harvard.edu",
    description="Toolbox for computational pathology",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy>=1.16.4,<1.20',
        'openslide-python',
        'javabridge',
        'python-bioformats',
        'pydicom',
        'h5py',
        'opencv-contrib-python',
        'python-spams',
        'matplotlib',
        'sklearn',
        'scikit-image',
        'dask',
        'anndata',
        'scanpy',
        'torch',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ]
)
