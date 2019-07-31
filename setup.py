import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pathml",
    version="0.0.1",
    author="Jacob Rosenthal",
    author_email="jacob_rosenthal@dfci.harvard.edu",
    description="Utilities for digital pathology",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Dana-Farber/pathml",
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy', 'skimage'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)
