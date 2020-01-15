#from distutils.core import setup
#from distutils.extension import Extension
from setuptools import setup
from setuptools.extension import Extension

import subprocess

import os

#update version
args = 'git describe --tags'
p = subprocess.Popen(args.split(), stdout=subprocess.PIPE)
version = p.communicate()[0].decode("utf-8").strip()

#### Versions
version = "0.1" # init

# Set this to true to add install_requires to setup
if True:
    install_requires=[
         'astropy>=2.0.0',
         'photutils>=0.3.0',
         'lxml>=3.8.0',
         'numpy>=1.10.2',
         'matplotlib>=2.0.2']
else:
    install_requires = []    
    
#lines = open('grizli/version.py').readlines()
version_str = """# git describe --tags
__version__ = "{0}"\n""".format(version)
fp = open('golfir/version.py','w')
fp.write(version_str)
fp.close()
print('Git version: {0}'.format(version))

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "golfir",
    version = version,
    author = "Gabriel Brammer",
    author_email = "gbrammer@gmail.com",
    description = "Great Observatories Legacy Fields: IR tools",
    license = "MIT",
    url = "https://github.com/gbrammer/golfir",
    download_url = "https://github.com/gbrammer/golfir/tarball/{0}".format(version),
    packages=['golfir', 'golfir/vlt'],
    classifiers=[
        "Development Status :: 1 - Planning",
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Astronomy',
    ],
    install_requires=install_requires,
    package_data={'golfir': ['data/*header']},
)
