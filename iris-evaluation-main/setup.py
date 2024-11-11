import os
import sys
from setuptools import setup, find_packages
from setuptools.command.install import install

setup(
    name='iris_evaluation',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.22.4',
        'opencv-contrib-python>=4.6.0.66',
        'opencv-python>=4.6.0.66',
        'pandas>=2.2.1',
        'polars>=1.8.2',
        'torch>=2.3.1',
        'scipy>=1.10.1',
        'seaborn>=0.13.2',
        'matplotlib>=3.5.2'
    ],
    author='Rob Hart',
    author_email='robjhart@iu.edu',
    description='A package for evaluating iris images.',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10.4',
)