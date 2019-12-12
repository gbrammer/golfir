.. image:: docs/_static/golfir_logo.png

``golfir``: Great Observatories Legacy Fields IR Analysis Tools
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This repository provides tools for modeling Spitzer IRAC and MIPS images based on high resolution templates from existing Hubble imaging in the context of the CHArGE / GOLF project (Brammer, Stefanon et al.).

Requirements: 
~~~~~~~~~~~~~
    .. code:: bash
    
       grizli
       astropy
       drizzlepac
       skimage
       ...
       
Installation:
~~~~~~~~~~~~~
    .. code:: bash
    
        $ git clone git@github.com:gbrammer/golfir.git
        $ cd golfir
        $ pip install . 
        
Usage:
~~~~~~
See the examples in the `notebooks` subdirectory:

IRAC-mosaic.ipynb - Generate drizzled IRAC mosaics and PSFs from individual Spitzer `BCD` exposure files.
 
