"""
End-to-end test of the modeler
"""
import os
import numpy as np

from .. import model


def test_full_patch():
    
    try:
        import grizli
    except:
        return True
    
    TEST_DIR = os.path.join(os.path.dirname(model.__file__), 'tests/Patch')
    if not os.path.exists(TEST_DIR):
        # Skip test if test data not found
        return True
        
    os.chdir(TEST_DIR)
    os.system('tar xzvf test_dataset_j213512m0103.tar.gz')
    
    root = 'j213512m0103'
    
    kwargs = {'ds9': None, 
              'patch_arcmin': 1.0,      # Size of patch to fit
              'patch_overlap': 0.2,     # Overlap of automatic patches
              'mag_limit': [24, 27],    # Two-pass modeling.  Fit sources brighter than mag_limit in HST catalog
              'run_alignment': True,    # Run fine alignment between IRAC-HST, in between two steps of `mag_limit`
              'galfit_flux_limit': np.inf,  # Brightness limit (uJy) of objects to fit with GALFIT
              'refine_brightest': True, # Refine masked bright objects with galfit
              'any_limit': 15,          # Brightness limit below which to mask *any* sources
              'point_limit': 15,        # Brightness limit below which to mask point-like sources
              'bright_sn': 30,          # S/N threshold for masked pixels of bright object
              'bkg_kwargs': {'order_npix': 64},          # Arguments to the local background routine
              'channels': ['ch1', 'ch2'],  # Channels to try
              'psf_only': False,        
              'use_saved_components': False, # Use models from a "components" file if found
              'window': None            # PSF-match windowing
              }
              
    # Target center
    ra, dec = 323.7882262, -1.050061227
    
    # Fit 0.6' square patch around the target of interest
    kwargs['patch_arcmin'] = 0.6
    
    # Run it
    ch = 'ch1'
                
    # Initialize
    modeler = model.ImageModeler(root=root, lores_filter=ch, **kwargs) 
        
    # Run the model
    modeler.run_full_patch(rd_patch=(ra, dec), **kwargs)
    
    return modeler
    
    