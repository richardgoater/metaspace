from setuptools import setup, find_packages

from sm import __version__

setup(name='sm',
      version=__version__,
      description='High throughput molecules annotation for imaging mass spectrometry data sets',
      url='https://github.com/SpatialMetabolomics/sm-engine.git',
      author='Alexandrov Team, EMBL',
      author_email='vitaly.kovalev@embl.de',
      packages=find_packages(),
      install_requires=[
          "pyImagingMSpec==0.1.4",
          "cpyImagingMSpec==0.2.4",
          "pyMSpec==0.1.2",
          "cpyMSpec==0.3.5",
          "pyimzML==1.2.1",
          "Pillow==4.2.1"
      ])
