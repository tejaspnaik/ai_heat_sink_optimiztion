"""
setup.py - Package installation script
"""
from setuptools import setup, find_packages

setup(
    name='pinn-cooling',
    version='0.1.0',
    description='Physics-Informed Neural Networks for Liquid Cooling Heat Sink Optimization',
    author='Your Name',
    license='MIT',
    packages=find_packages(),
    install_requires=[
        'torch>=2.0.0',
        'numpy>=1.24.0',
        'scipy>=1.10.0',
        'matplotlib>=3.7.0',
        'pyyaml>=6.0',
        'tqdm>=4.65.0'
    ],
    extras_require={
        'optimization': ['optuna>=3.0.0', 'pymoo>=0.6.0', 'scikit-optimize>=0.9.0'],
        'viz': ['plotly>=5.0.0', 'pandas>=2.0.0'],
        'data': ['h5py>=3.9.0', 'xarray>=2023.0.0', 'netCDF4>=1.6.0'],
        'notebook': ['jupyter>=1.0.0'],
    },
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Physics',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11'
    ]
)
