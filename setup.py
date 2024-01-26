#!/usr/bin/env python
from setuptools import setup

setup(name='eprTools',
      version='0.1',
      description='python tools for EPR',
      author='Maxx Tessmer',
      author_email='mhtessmer@gmail.com',
      install_requires=['numpy', 'scipy', 'scikit-learn', 'matplotlib', 'numba', 'cvxopt'],
      packages=['eprTools'])
