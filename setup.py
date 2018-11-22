# -*- coding: utf-8 -*-
import os
from setuptools import find_packages, setup

with open(os.path.join(os.path.dirname(__file__), 'README.md'), encoding='UTF-8') as readme:
    README = readme.read()

# allow setup.py to be run from any path
os.chdir(os.path.normpath(os.path.join(os.path.abspath(__file__), os.pardir)))

setup(
    name='scan_ts',
    version='0.1',
    packages=find_packages(),
    include_package_data=True,
    license='GPL License',  # example license
    description='Automated Relaxed Potential Energy Surface Scans',
    long_description=README,
    url='https://github.com/renpj/scan_ts',
    author='Ren',
    author_email='openrpj@gmail.com',
    classifiers=[
        'Intended Audience :: Scientists and Developers',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ], 
    install_requires=['ase']
)