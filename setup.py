from setuptools import setup

# read the contents of your README file
from os import path

with open('README.md') as f:
    long_description = f.read()


setup(
    name='covid19_inference',
    author='Jonas Dehning, Johannes Zierenberg, F. Paul Spitzner, Michael Wibral, Joao Pinheiro Neto, Michael Wilczek, Viola Priesemann',
    author_email='jonas.dehning@ds.mpg.de',
    packages=['covid19_inference'],
    url='https://github.com/Priesemann-Group/covid19_inference_forecast',
    python_requires='>=3.6.0',
    version = '0.0.3'
)
