from setuptools import setup
from setuptools import find_packages

setup(name = 'ModGNN',
      version = '0.0.1',
      packages = find_packages(),
      install_requires = ['torch'],
      author = 'Ryan Kortvelesy',
      author_email = 'rk627@cam.ac.uk',
      description = 'A modular GNN framework',
      long_description = 'A modular GNN framework developed by the Prorok Lab at the University of Cambridge'
)