#!/usr/bin/env python
from distutils.core import setup

setup(name = 'eprTools',
      version = '0.1',
      description = 'python tools for EPR',
      author = 'Maxx Tessmer',
      author_email = 'mhtessmer@gmail.com',
      install_requires = ['numpy', 'scipy', 'sklearn', 'matplotlib', 'numba'],
      packages=['eprTools'])
      
