"""Installation script for SmartStock.
"""
import os
from setuptools import setup, find_packages

# Path to current directory.
CWD = os.path.abspath(os.path.dirname(__file__))

def get_install_requires():

    # Parse dependencies from requirements file.
    # This is useful for Docker-based installs.
    with open(os.path.join(CWD, 'requirements.txt')) as f:
        return f.read().splitlines()

def get_description():

    # Obtain long description from README file.
    with open(os.path.join(CWD, 'README.md')) as f:
        return f.read()

setup(
    name='smart-stock',
    version='0.0.1',
    description='SmartStock Reinforcement Learning',
    long_description=get_description(),
    install_requires=get_install_requires(),
    extras_require={
        'test': [
            'pytest',
            'stable-baselines3',
        ],
    },
    packages=find_packages(exclude='tests'),
    author='Alexander DeRieux',
    author_email='alexander.derieux@gmail.com',
    url='https://github.com/zanderman/smart-stock-rl',
)