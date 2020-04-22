#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import warnings
from setuptools import setup


ROOT = os.path.join(os.path.dirname(__file__), 'SimMultiTrans')
CONFIG = os.path.join(ROOT, 'conf')
RESULTS = os.path.join(ROOT, 'results')

try:
    os.makedirs(CONFIG, exist_ok=True)
    os.makedirs(RESULTS, exist_ok=True)
except OSError:
    pass

empty_sts = 0

try:
    empty_sts = os.path.getsize(os.path.join(CONFIG, '.mapbox_token'))
except OSError:
    warnings.warn('Missing token file in ./conf')

if empty_sts == 0:
    f = open(os.path.join(CONFIG, '.mapbox_token'), 'w')
    token = input('Paste your Mapbox token and press Enter: ')
    f.write(token)
    f.close()

try:
    empty_sts = os.path.getsize(os.path.join(CONFIG, '.mapbox_style'))
except OSError:
    warnings.warn('Missing style file in ./conf')

if empty_sts == 0:
    f = open(os.path.join(CONFIG, '.mapbox_style'), 'w')
    style_key = input('Paste your Mapbox style URL and press Enter: ')
    f.write(style_key)
    f.close()


setup(name='SimMultiTrans',
      version='0.1',
      description='Simulator',
      author='momodupi',
      author_email='momodupi@gmail.com',
      license='MIT',
      url='https://github.com/haochen3611/SimMultiTrans',
      packages=['SimMultiTrans', ],
      install_requires=['numpy',
                        'pandas',
                        'matplotlib',
                        'scipy',
                        'plotly',
                        'ipython'],
      zip_safe=False)
