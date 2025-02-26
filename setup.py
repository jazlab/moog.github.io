"""Installation script."""

from setuptools import find_packages
from setuptools import setup
import os

this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='moog-games',
    version='1.6',
    description=(
        'Modular object oriented games utils is a python-based game engine.'),
    author='Nicholas Watters',
    url='https://github.com/jazlab/moog.github.io',
    license='MIT license',
    long_description=long_description,
    long_description_content_type='text/markdown',
    keywords=[
        'ai',
        'reinforcement-learning',
        'python',
        'machine learning',
        'objects',
        'psychology',
        'neurophysiology',
        'psychophysics',
        'physics',
        'environment',
    ],
    packages=(
        ['moog_demos', 'moog'] + 
        ['moog_demos.' + x for x in find_packages('moog_demos')] +
        ['moog.' + x for x in find_packages('moog')]
    ),
    install_requires=[
        'cmake'
        'numpy==2.2.3',
        'matplotlib==3.10.0',
        'imageio==2.37.0',
        'mss==10.0.0',
        'tqdm==4.67.1',
        'dm_env==1.6',
        'gym==0.26.2',
    ],
    tests_require=[
        'nose==1.3.7',
        'pytest==8.3.4',
    ],
    test_suite='nose.collector',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: POSIX :: Linux',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: MacOS :: MacOS X',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering',
    ],
)