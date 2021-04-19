"""
Test golfir utils
"""

import numpy as np

from .. import utils

def test_convolve():
    """
    Test 2D convolution helper
    """
    
    psf = np.ones((16,16))
    psf /= 16**2
    
    img = np.zeros((129,129))
    img[64, 64] = 1.
    
    for method in ['fftconvolve','xstsci','oaconvolve']:
        test = utils.convolve_helper(img, psf, method=method, 
                                     fill_scipy=False)
    
        assert (np.allclose(test.shape, img.shape))
        assert (np.allclose(test[64,64]*16**2, 1.0, rtol=1.e-4))
        
        test = utils.convolve_helper(img, psf, method=method, 
                                     fill_scipy=True, cval=0)
        
        assert (np.allclose(test[0,0], 0., rtol=1.e-4))
        
        test = utils.convolve_helper(img, psf, method=method, 
                                     fill_scipy=True, cval=1)
        
        assert (np.allclose(test[0,0], 0.75, rtol=1.e-4))
        assert (np.allclose(test[0,64], 0.5, rtol=1.e-4))
        assert (np.allclose(test[64,0], 0.5, rtol=1.e-4))
        