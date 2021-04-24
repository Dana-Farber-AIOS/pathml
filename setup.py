import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pathml",
    version="0.0.1",
    author="Jacob Rosenthal",
    author_email="Jacob_Rosenthal@dfci.harvard.edu",
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
        'dask'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ]
)
