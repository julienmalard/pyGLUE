import os
from setuptools import setup

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "pyGLUE",
    version = "0.0.4",
    author = "Joost Delsman",
    author_email = "joostdelsman@gmail.com",
    description = ("A Python framework to conduct GLUE analyses"),
    license = "BSD",
    keywords = "GLUE hydrology uncertainty model modeling modelling",
    url = "http://packages.python.org/pyGLUE",
    packages=['pyGLUE'],
    long_description=read('README'),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Scientific/Engineering",
        "License :: OSI Approved :: BSD License",
    ],
)