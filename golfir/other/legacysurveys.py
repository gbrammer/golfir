"""
Prepare images from legacysurveys.org for use with GOLFIR
"""
import glob
import os

import numpy as np

import astropy.io.fits as pyfits
import astropy.wcs as pywcs

def image_prep(root='j112716p4228', brick='1717p425', pad=16):
    """
    Prepare image mosaics
    """

    hst_files = glob.glob(f'{root}-[fi][1r]*sci.fits*')
    lfiles = glob.glob(f'*-{brick}*image*fits.fz')
    
    hst_file = pyfits.open(hst_files[0])
    hst_wcs = pywcs.WCS(hst_file[0].header)
    hst_fp = hst_wcs.calc_footprint()
    
    for file in lfiles:
        if '_drz' in file:
            ext = 0
        else:
            ext = 1
            
        im = pyfits.open(file)[ext]
        wht_file = file.replace('image', 'invvar').replace('_sci.f','_wht.f')
        wht = pyfits.open(wht_file)[ext]
        wcs = pywcs.WCS(im.header)
        hst_pix = wcs.all_world2pix(hst_fp, 0)
        
        # Padding
        sh = im.data.shape
        ll = hst_pix[0,:]
        ur = hst_pix[2,:]
        
        left = -np.cast[int](np.maximum(-(ll-pad), 0))
        right = np.cast[int](np.maximum(ur+pad-np.array(sh[::-1]), 0))
        extra = left + right
        
        head = im.header.copy()
        sci_data = np.zeros((sh[0]+extra[1], sh[1]+extra[0]), dtype=np.float32)
        sci_data[left[1]:left[1]+sh[0], left[0]:left[0]+sh[1]] += im.data
        wht_data = np.zeros((sh[0]+extra[1], sh[1]+extra[0]), dtype=np.float32)
        wht_data[left[1]:left[1]+sh[0], left[0]:left[0]+sh[1]] += wht.data
        
        head['CRPIX1'] += left[0]
        head['CRPIX2'] += left[1]
        
        # Scale to AB
        if 'BUNIT' in head:
            if head['BUNIT'] == 'nanomaggy':
                scl = 10**(-0.4*(22.5-23.9))
                head['PHOTFNU'] = 1.e-6  
            else:
                scl = 1.
                head['PHOTFNU'] = 1.
        else:
            scl = 1.
            
        sci_data *= scl
        wht_data *= 1./scl**2
        
        if 'FILTERX' in head:
            filt = head['FILTERX'] + head['INSTRUME']
        else:
            filt = head['FILTER'].lower()
            
        pyfits.PrimaryHDU(data=sci_data, header=head).writeto(f'{root}-{filt}_drz_sci.fits', overwrite=True)
        pyfits.PrimaryHDU(data=wht_data, header=head).writeto(f'{root}-{filt}_drz_wht.fits', overwrite=True)

def make_psf(root='j112716p4228', psf_rd=None, ds9=None, N=32):
    """
    Make the image PSF
    """
    import matplotlib.pyplot as plt
    
    from grizli import utils
    from golfir.irac import MixturePSF  
    
    files = glob.glob(f'{root}-[grz]*sci.fits')
    files.sort()
    
    if (psf_rd is None) & (ds9 is not None):
        psf_rd = np.cast[float](ds9.get('pan fk5').split())
    
    obj = MixturePSF(N=5) 
    
    for file in files:
        im = pyfits.open(file)
        wcs = pywcs.WCS(im[0].header)
        xy = np.cast[int](np.round(np.array(wcs.all_world2pix(np.atleast_2d(psf_rd), 0)).flatten()))
        slx = slice(xy[0]-N, xy[0]+N)
        sly = slice(xy[1]-N, xy[1]+N)
        
        psf = im[0].data[sly, slx]        
        for iter in range(10):
            fig = obj.from_cutout(psf, mask=None, show=(iter == 9), center_first=True)
        
        plt.gcf().savefig(file.replace('_drz_sci.fits', '_psf.png'))
        
        obj.recenter()
        
        h = utils.get_wcs_slice_header(wcs, slx, sly)
    
        pars = obj.getParams()
        h['PSFN'] = (obj.N, 'MixturePSF N components')
        for i in range(obj.N):
            h[f'PSFC{i}'] = obj.coeffs[i]
            pars = obj.mogs[i].getParams()
            for j in range(6):
                h[f'PSFP{i}_{j}'] = pars[j]
                
        pyfits.PrimaryHDU(data=psf, header=h).writeto(file.replace('_drz_sci', '_psf'), overwrite=True)


def run_model(root=''):
    
    import golfir.irac
    import golfir.model
    
    from astropy.modeling.models import Moffat2D, Gaussian2D, Sersic2D
    from astropy.modeling.fitting import LevMarLSQFitter   
    
    from tractor.psf import GaussianMixtureEllipsePSF, GaussianMixturePSF
    
    from golfir.irac import MixturePSF  
    
    from photutils import (HanningWindow, TukeyWindow, CosineBellWindow, SplitCosineBellWindow, TopHatWindow)
    window = CosineBellWindow(1) 
    
    fitter = LevMarLSQFitter()
    
    import grizli.ds9
    ds9 = grizli.ds9.DS9()
    
    P0 = None
    bkg_func = None
    
    kwargs = {'ds9':ds9, 'mag_limit':[24,27], 'galfit_flux_limit':np.inf, 'refine_brightest':False, 'run_alignment':True, 'any_limit':10, 'point_limit':-10, 'bright_sn':10, 'bkg_kwargs':{'order_npix':64}, 'psf_only':False, 'use_saved_components':False, 'window':None, 'use_avg_psf':True, 'align_type':1} 
    
    files = glob.glob(f'{root}-[kgrz]*sci.fits')
    files.sort()
    
    bands = [file.split('_drz')[0].split('-')[-1] for file in files]
    
    orig_pix = 0.262
    
    for band in bands:
        psf_im = pyfits.open(f'{root}-{band}_psf.fits')[0]
        h = psf_im.header

        psf_obj = MixturePSF(N=psf_im.header['PSFN']) 
        
        for i in range(psf_obj.N):
            psf_obj.coeffs[i] = h[f'PSFC{i}']
            pars = [h[f'PSFP{i}_{j}'] for j in range(6)]
            psf_obj.mogs[i].setParams(pars)

        psf_obj.set_pixelgrid(size=32, instep=orig_pix, outstep=orig_pix, oversample=orig_pix/0.1)
        
        modeler = golfir.model.ImageModeler(root=root, prefer_filter='f160w', 
                                            lores_filter=band, 
                                            psf_obj=psf_obj) 
                                            
        if not os.path.exists(f'{root}_waterseg.fits'):
            pyfits.writeto(f'{root}_waterseg.fits', data=modeler.waterseg)

def run_photoz(root='', k_hawki=True):
    # xxxx
    from grizli import utils
    from grizli.pipeline import photoz
    import numpy as np
    
    if k_hawki:
        kfilt = '269' # HAWKI    
    else:
        kfilt = '259' # VISTA    
    
    extra_translate = {'HSCg_flux':'F314', 
                       'HSCg_err' :'E314',
                       'HSCr_flux':'F315', 
                       'HSCr_err' :'E315',
                       'HSCi_flux':'F316', 
                       'HSCi_err' :'E316',
                       'HSCz_flux':'F317', 
                       'HSCz_err' :'E317',
                       'HSCy_flux':'F318', 
                       'HSCy_err' :'E318',
                       'HSCn816_flux':'F319', 
                       'HSCn816_err' :'E319',
                       'ks_flux': 'F'+kfilt,
                       'ks_err' : 'E'+kfilt, 
                       'zMosaic3_flux_aper': 'F297', # DECam z
                       'zMosaic3_fluxerr_aper':  'E297',
                       'r90prime_flux_aper': 'F295', # DECam r
                       'r90prime_fluxerr_aper':  'E295'}
    
    if True:
        extra = {'TEMPLATES_FILE':'templates/fsps_full_2019/xfsps_QSF_12_v3.SB.param'}

        new_wave = np.hstack([utils.log_zgrid([100, 2.e4], 1000./3.e5), utils.log_zgrid([2.e4, 1.e8], 1e4/3.e5)])
        extra['RESAMPLE_WAVE'] = new_wave
        extra['SYS_ERR'] = 0.03
    else:
        extra = {}

    self, cat, zout = photoz.eazy_photoz(root+'_irac', object_only=False, apply_prior=False, beta_prior=True, aper_ix=1, force=True, get_external_photometry=False, compute_residuals=False, total_flux='flux_auto', extra_params=extra, extra_translate=extra_translate, zpfile='zphot.zeropoint')
    
    return self, cat, zout
    
    