# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

# Get the long description from the README file
with open('README.md', 'r') as f:
    long_description = f.read()

setup(name='axel_lab_to_nwb',
      version='0.0.1',
      description='NWB conversion scripts and tutorials.',
      long_description=long_description,
      long_description_content_type='text/markdown',
      author='Luiz Tauffer and Ben Dichter',
      email='ben.dichter@gmail.com',
      packages=find_packages(),
      include_package_data=True,
      install_requires=['matplotlib', 'cycler', 'scipy', 'numpy', 'jupyter',
                        'h5py', 'pynwb', 'ndx-grayscalevolume'],
      )
