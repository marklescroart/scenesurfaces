from setuptools import setup, find_packages

requirements = [
    'numpy', 'scipy', 'couchdb', ] # file_io, docdb_lite

setup(
    name='scenesurfaces',
    version='0.0.2',
    packages=find_packages(),
    long_description=open('README.md').read(),
    install_requires=requirements,
)
