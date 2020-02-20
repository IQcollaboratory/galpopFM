#!/usr/bin/env python
from setuptools import setup

__version__ = '0.1'

setup(name = 'galpopfm',
      version = __version__,
      python_requires='>3.5.2',
      description = 'forward modeling galaxy populations',
      install_requires = ['numpy', 'matplotlib', 'h5py', 'abcpmc'], #'mpi4py', 
      provides = ['galpopfm'],
      packages = ['galpopfm'],
      include_package_data=True, 
      package_data={'galpopfm': ['dat/*.dat']}
      )
