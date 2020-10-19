#!/usr/bin/env python
from setuptools import setup

__version__ = '0.1'

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name = 'galpopfm',
        version = __version__,
        author="ChangHoon Hahn", 
        author_email="hahn.changhoon@gmail.com", 
        description = 'Python Package for forward modeling galaxy populations from simulations',
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/IQcollaboratory/galpopFM/tree/master/galpopfm", 
        packages=setuptools.find_packages(),
        install_requires = ['numpy', 'matplotlib', 'h5py', 'abcpmc', 'corner'], #'mpi4py', 
        provides = ['galpopfm'],
        include_package_data=True, 
        package_data={'galpopfm': ['dat/*.dat']},
        python_requires='>3.6'
        )
