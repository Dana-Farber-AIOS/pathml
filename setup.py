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
        'numpy',
        'openslide-python',
        'opencv-contrib-python',
        'matplotlib',
        'sklearn',
        'dask',
        'anndata',
        'pydicom'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ]
)
