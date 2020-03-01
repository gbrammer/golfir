"""
Image-based modeling of IRAC fields
"""
import os
import glob
from collections import OrderedDict
import traceback

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path

import scipy.ndimage as nd
import scipy.spatial
from scipy.optimize import minimize

from stsci.convolve import convolve2d
from skimage.transform import warp
from skimage.transform import SimilarityTransform
from skimage.morphology import dilation
from skimage.morphology import binary_dilation

import astropy.io.fits as pyfits
import astropy.wcs as pywcs
import astropy.units as u

from photutils import create_matching_kernel
from photutils import (HanningWindow, TukeyWindow, CosineBellWindow,
                       SplitCosineBellWindow, TopHatWindow)

import drizzlepac
from drizzlepac.astrodrizzle import ablot

import grizli.utils

from tqdm import tqdm

# from golfir.utils import get_wcslist, _obj_shift
# from golfir import irac

class model_psf(object):
    def __init__(self, psf_object, transform=None):
        self.psf_object = psf_object
        self.transform = transform

        # Windowing for PSF match
        window = HanningWindow()
        
class ImageModeler(object):
    
    def __init__(self, root='j132352p2725', prefer_filter='f160w', lores_filter='ch1', verbose=True, psf_only=False, **kwargs):
        self.root = root
        self.LOGFILE=f'{root}.modeler.log.txt'
        self.verbose = verbose
        
        self.patch_rd = (0., 0.)
        self.patch_npix = 0
        self.patch_ids = []
        self.patch_nobj = 0
                        
        self.psf_window = -1
        self.psf_only = psf_only
        
        # Read High-res (Hubble) data
        self.read_hst_data(prefer_filter=prefer_filter)
        
        # Dilate seg image
        self.watershed_segmentation()
        
        # Read Low-res (IRAC) data
        self.read_lores_data(filter=lores_filter)
        
        # Initialize PSFs
        self.init_psfs(**kwargs)
        
        # Component file
        component_file = f'{self.root}-{self.lores_filter}_components.fits'
        if os.path.exists(component_file):
            grizli.utils.log_comment(self.LOGFILE, 
                   f'Read component file "{component_file}".', 
                   verbose=self.verbose, show_date=True)
                   
            self.comp_hdu = pyfits.open(component_file)
            self.full_mask = self.comp_hdu[0].data
        else:
            self.full_mask = np.zeros(self.lores_im.data.shape,
                                      dtype=np.uint8)
            prime = pyfits.PrimaryHDU(header=self.lores_im.header, 
                                      data=self.full_mask)
            self.comp_hdu = pyfits.HDUList([prime])
            self.comp_hdu[0].header['EXTNAME'] = 'MASK'
        
        # testing
        if False:
            bright = [12,16]
            bright = [18,18]
            bright_args = dict(any_limit=bright[0], point_limit=bright[1], point_flux_radius=3.5, bright_ids=None, bright_sn=7)
            mag_limit = 24
            
            reload(golfir.model)
            self = golfir.model.ImageModeler() 
            self.patch_initialize(rd_patch=None, patch_arcmin=0.6, ds9=ds9, patch_id=0)
            
            self.patch_compute_models(mag_limit=mag_limit, **bright_args)
            
            self.patch_least_squares()
            self.patch_display(ds9=ds9)
            
            self.patch_align()
            ds9.frame(15)
            ds9.view(self.patch_resid*self.patch_mask,
                     header=self.patch_header)
            
            # Regenerate models with alignment transform
            self.patch_compute_models(mag_limit=mag_limit, **bright_args)
            self.patch_least_squares()
            ds9.frame(16)
            ds9.view(self.patch_resid*self.patch_mask,
                     header=self.patch_header)
            
            self.patch_align()
            ds9.frame(17)
            ds9.view(self.patch_resid*self.patch_mask,
                     header=self.patch_header)
            
            # Bright star
            brighter_args = dict(any_limit=2, point_limit=2, point_flux_radius=3.5, bright_ids=None, bright_sn=7)
            self.patch_bright_limits(**brighter_args)
            
            self.patch_fit_galfit(ids=[1684], component_type='psf', chi2_threshold=10)
            ds9.frame(18)
            ds9.view(self.patch_resid*self.patch_mask,
                     header=self.patch_header)
            
            self.patch_fit_galfit(ids=[2019,2020], component_type=None, chi2_threshold=10)
            self.patch_fit_galfit(ids=[2235,2351], component_type=None, chi2_threshold=10)
            
    @staticmethod
    def fetch_from_aws(root):
        """
        Fetch data from AWS
        """
        
        os.system(f'aws s3 sync s3://grizli-v1/Pipeline/{root}/IRAC/ ./')
        os.system(f'aws s3 sync s3://grizli-v1/Pipeline/{root}/Prep/ ./'
                   ' --exclude "*"'
                   ' --include "j*-ir*_[sw]??.fits.gz"'
                   ' --include "j*-f1*_[sw]??.fits.gz"'
                   ' --include "*psf.fits" --include "j*seg.fits.gz"'
                   ' --include "j*phot.fits"')
        
        os.system('gunzip -f *drz*fits.gz *seg.fits.gz')
        
        # Need ACS?
        wfc_files = glob.glob('j*-f1*sci.fits')
        if len(wfc_files) == 0:
            os.system(f'aws s3 sync s3://grizli-v1/Pipeline/{root}/Prep/ ./'
                       ' --exclude "*"'
                       ' --include "j*-f[678]*_[sw]??.fits.gz"')

            os.system('gunzip -f *_dr*fits.gz *seg.fits.gz')
            
        if not os.path.exists(f'{root}_irac_phot.fits'):
            os.system(f'cp {root}_phot.fits {root}_irac_phot.fits')
            
    def read_hst_data(self, prefer_filter='f160w'):
        """
        Read HST data
        """
        ref_files = glob.glob('{0}-f1*_drz_sci.fits'.format(self.root))
        ref_files.sort()
        if len(ref_files) == 0:
            ref_files = glob.glob('{0}-ir*_dr*_sci.fits'.format(self.root))
            ref_files.sort()
            
        if f'{self.root}-{prefer_filter}_drz_sci.fits' in ref_files:
            ref_file = f'{self.root}-{prefer_filter}_drz_sci.fits'
        else:
            ref_file = ref_files[-1]
        
        grizli.utils.log_comment(self.LOGFILE, 
               f'Use HST reference image "{ref_file}".', verbose=self.verbose, 
               show_date=True)
        
        hst_im = pyfits.open(ref_file)
        
        ref_filter = grizli.utils.get_hst_filter(hst_im[0].header).lower()
        hst_wht = pyfits.open(ref_file.replace('_sci', '_wht'))
        
        try:
            hst_psf = pyfits.open('{0}-{1}_psf.fits'.format(self.root,
                                  ref_filter))['PSF','DRIZ1'].data
        except:
            hst_psf = pyfits.open('{0}-{1}_psf.fits'.format(self.root, 
                                  ref_filter))[1].data
                                  
        hst_psf /= hst_psf.sum()

        hst_seg = pyfits.open(self.root+'-ir_seg.fits')[0]
        hst_ujy = hst_im[0].data*hst_im[0].header['PHOTFNU'] * 1.e6
        hst_wcs = pywcs.WCS(hst_im[0].header)
        hst_wcs.pscale = grizli.utils.get_wcs_pscale(hst_wcs)

        phot = grizli.utils.read_catalog(f'{self.root}_irac_phot.fits')

        # Additional NNeighbor columns
        if ('id_nn' not in phot.colnames) | False:

            pos = np.array([phot['ra'], phot['dec']]).T
            pos = (pos - np.median(pos,  axis=0))
            tree = scipy.spatial.cKDTree(pos)
            dn, nn = tree.query(pos, k=2)
            phot['dr_nn'] = dn[:,1]*3600*u.arcsec
            phot['dr_nn'].format = '.2f'

            phot['dmag_nn'] = phot['mag_auto'] - phot['mag_auto'][nn[:,1]]
            phot['dr_nn'].format = '.2f'

            phot['id_nn'] = phot['number'][nn[:,1]]
        
        self.phot = phot
        
        self.ref_file = ref_file
        self.ref_filter = ref_filter

        self.hst_im = hst_im
        self.hst_ujy = hst_ujy

        self.hst_wht = hst_wht
        self.hst_seg = hst_seg
        
        self.hst_psf = hst_psf
        self.hst_wcs = hst_wcs
    
    def watershed_segmentation(self, smooth_size=5):
        """
        Dilate the SExtractor/SEP segmentation image using the 
        `~skimage.morphology.watershed` algorithm.
        """
        from skimage.morphology import watershed
        from astropy.convolution.kernels import Gaussian2DKernel
        
        grizli.utils.log_comment(self.LOGFILE, 
               f'Watershed segmentation dilation, smooth_size={smooth_size}',
               verbose=self.verbose, show_date=True)
        
        # globaal median
        med = np.median(self.hst_im[0].data[(self.hst_wht[0].data > 0) 
                         & (self.hst_seg.data == 0)])
        
        # Convolved with Gaussian kernel                 
        kern = Gaussian2DKernel(smooth_size).array
        hst_conv = convolve2d(self.hst_im[0].data-med, kern, fft=1)
        hst_var = 1/self.hst_wht[0].data
        wht_med = np.percentile(self.hst_wht[0].data[self.hst_wht[0].data > 0], 5)
        hst_var[self.hst_wht[0].data < wht_med] = 1/wht_med
        hst_cvar = convolve2d(hst_var, kern**2, fft=1)

        xi = np.cast[int](np.round(self.phot['xpeak']))
        yi = np.cast[int](np.round(self.phot['ypeak']))
        markers = np.zeros(self.hst_im[0].data.shape, dtype=int)
        markers[yi, xi] = self.phot['number']

        waterseg = watershed(-hst_conv, markers, mask=(hst_conv/np.sqrt(hst_cvar) > 1))
        waterseg[waterseg == 0] = self.hst_seg.data[waterseg == 0]
        self.waterseg = waterseg
        
    def read_lores_data(self, filter='ch1', use_avg_psf=True, psf_obj=None):
        """
        Read low-res (e.g., IRAC)
        """
        from golfir import irac
        
        grizli.utils.log_comment(self.LOGFILE, 
               f'Read lores data: {self.root}-{filter}\n',
               verbose=self.verbose, show_date=True)
        
        ################
        # Parameters for IRAC fitting
        if use_avg_psf & (filter in ['ch1','ch2','ch3','ch4']):
            #avg_psf = pyfits.open('{0}-{1}-0.1.psfr_avg.fits'.format(self.root, filter))[0].data
            _path = os.path.dirname(irac.__file__)
            
            if 0:
                # My stacked psfs
                psf_file = os.path.join(_path, 
                                    f'data/psf/irac_{filter}_cold_psf.fits')
            else:
                # Generated from IRSA PRFs
                psf_file = os.path.join(_path, 
                                    f'data/psf/irsa_0.1pix_{filter}_psf.fits')
            
            avg_psf = pyfits.open(psf_file)[0].data
            
            grizli.utils.log_comment(self.LOGFILE, 
                   f'Read avg psf: {psf_file}\n',
                   verbose=self.verbose, show_date=True)
                   
            self.avg_psf_file = psf_file
        else:
            avg_psf = None
            self.avg_psf_file = None

        avg_kern = None #np.ones((5,5))
        
        self.galfit_parameters = OrderedDict()
        
        ###############
        # Load Spitzer images
        if filter == 'mips1':
            irac_im = pyfits.open('{0}-{1}_drz_sci.fits'.format(self.root, filter))[0]
            irac_wht = pyfits.open('{0}-{1}_drz_wht.fits'.format(self.root, filter))[0].data
            irac_psf_obj = irac.MipsPSF()
            pf = 10 # assume 1" pixels

            self.column_root = 'mips_24'

            #ERR_SCALE = 0.1828 # From residuals
            self.ERR_SCALE = 1.

        elif filter in ['ch1','ch2','ch3','ch4']:
            irac_im = pyfits.open('{0}-{1}_drz_sci.fits'.format(self.root, filter))[0]
            irac_wht = pyfits.open('{0}-{1}_drz_wht.fits'.format(self.root, filter))[0].data
            irac_psf_obj = irac.IracPSF(ch=int(filter[-1]), scale=0.1, verbose=self.verbose, avg_psf=avg_psf)

            self.ERR_SCALE = 1.
            # if os.path.exists('{0}-{1}.cat.fits'.format(self.root, filter)):
            #     ir_tab = grizli.utils.read_catalog('{0}-ch{1}.cat.fits'.format(self.root, filter))
            #     if 'ERR_SCALE' in ir_tab.meta:
            #         self.ERR_SCALE = ir_tab.meta['ERR_SCALE']

            # Integer ratio of pixel sizes between IRAC and HST
            try:
                irac_wcs = pywcs.WCS(irac_im.header)
                irac_wcs.pscale = utils.get_wcs_pscale(irac_wcs)
                pf = int(np.round(irac_wcs.pscale/self.hst_wcs.pscale))
            except:
                pf = 5

            self.column_root = 'irac_{0}'.format(filter)        
            
            # exptime keywords for IRAC
            nexp, expt = irac_psf_obj.get_exposure_time(self.phot['ra'], 
                                        self.phot['dec'], verbose=True)
                                        
            self.phot[f'{self.column_root}_nexp'] = nexp
            self.phot[f'{self.column_root}_exptime'] = expt
        
        else:
            
            irac_im = pyfits.open('{0}-{1}_drz_sci.fits'.format(self.root, filter))[0]
            irac_wht = pyfits.open('{0}-{1}_drz_wht.fits'.format(self.root, filter))[0].data
            self.ERR_SCALE = 1.
            self.column_root = filter
            
            irac_wcs = pywcs.WCS(irac_im.header)
            irac_wcs.pscale = utils.get_wcs_pscale(irac_wcs)
            pf = int(np.round(irac_wcs.pscale/self.hst_wcs.pscale))
            
        if f'{self.column_root}_flux' not in self.phot.colnames:
            self.phot[f'{self.column_root}_flux'] = -99.
            self.phot[f'{self.column_root}_err'] = -99.
            self.phot[f'{self.column_root}_patch'] = 0
            self.phot[f'{self.column_root}_bright'] = 0
        
        if psf_obj is None:
            self.lores_psf_obj = irac_psf_obj
        else:
            self.lores_psf_obj = psf_obj
            
        self.lores_im = irac_im
        self.lores_shape = self.lores_im.data.shape
        self.lores_wht = irac_wht
        self.lores_wcs = irac_wcs
        self.lores_xy = irac_wcs.all_world2pix(self.phot['ra'], 
                                               self.phot['dec'], 0)
        self.lores_xy = np.array(self.lores_xy).T
        self.pf = pf
        self.lores_filter = filter
        
        # HST weight in LoRes frame
        self.hst_wht_i = self.hst_wht[0].data[::self.pf, ::self.pf]
        corner = self.hst_wcs.all_pix2world(np.array([[-0.5, -0.5]]), 0)
        ll = np.cast[int](self.lores_wcs.all_world2pix(corner, 0)).flatten()
        self.lores_mask = np.zeros(self.lores_shape)
        
        isly = slice(ll[1], ll[1]+self.hst_wht_i.shape[0])
        islx = slice(ll[0], ll[0]+self.hst_wht_i.shape[1])
        self.lores_wht[isly, islx] *= self.hst_wht_i > 0
        
        self.lores_cutout_sci = None
        
    def init_psfs(self, window=None, **kwargs):#HanningWindow()):
        
        ### (Restart from here after computing alignment below)

        ##################
        # HST PSF in same dimensions as IRAC
        self.lores_tf = None
        rd = self.hst_wcs.wcs.crval
        _psf, _, _ = self.lores_psf_obj.evaluate_psf(ra=rd[0], dec=rd[1], 
                                      min_count=1, clip_negative=True, 
                                      transform=self.lores_tf)

        hst_psf_full = np.zeros_like(_psf)
        
        hst_psf_offset = [2,2]
        hst_psf_size = self.hst_psf.shape[0]//2
        
        sh_irac = hst_psf_full.shape
        hslx = slice(sh_irac[0]//2+hst_psf_offset[0]-hst_psf_size, sh_irac[0]//2+hst_psf_offset[0]+hst_psf_size)
        hsly = slice(sh_irac[0]//2+hst_psf_offset[1]-hst_psf_size, sh_irac[0]//2+hst_psf_offset[1]+hst_psf_size)
        hst_psf_full[hsly, hslx] += self.hst_psf
        
        self.hst_psf_full = hst_psf_full
    
        self.psf_window = window
        
    @property
    def lores_sivar(self):
        if np.isfinite(self.ERR_SCALE):
            return np.sqrt(self.lores_wht)/self.ERR_SCALE
        else:
            return np.sqrt(self.lores_wht)
            
    def patch_initialize(self, rd_patch=None, patch_arcmin=1.4, ds9=None, patch_id=0, bkg_kwargs={}):
                
        self.patch_arcmin = patch_arcmin
        self.patch_npix = int(patch_arcmin*60*10/self.pf)
        self.patch_id = patch_id
        
        # Centered on HST
        if rd_patch is None:
            if ds9:
                rd_patch = np.cast['float'](ds9.get('pan fk5').split())
            else:
                rd_patch = self.hst_wcs.wcs.crval

        self.patch_rd = rd_patch
        
        grizli.utils.log_comment(self.LOGFILE, 
               f'Get {patch_arcmin}\' patch around '
               f'({rd_patch[0]:.5f}, {rd_patch[1]:.5f})\n',
               verbose=self.verbose, show_date=True)
                
        xy_lores = np.cast[int](np.round(self.lores_wcs.all_world2pix(np.array([rd_patch]), 0))).flatten()

        ll_lores = xy_lores - self.patch_npix
        corner = self.lores_wcs.all_pix2world(np.array([ll_lores])-0.5, 0)
        ll_hst_raw = self.hst_wcs.all_world2pix(corner, 0)
        ll_hst = np.cast[int](np.ceil(ll_hst_raw)).flatten()

        slx = slice(ll_hst[0], ll_hst[0]+2*self.patch_npix*self.pf)
        sly = slice(ll_hst[1], ll_hst[1]+2*self.patch_npix*self.pf)
        
        self.hst_slx, self.hst_sly = slx, sly
        self.patch_seg = self.waterseg[sly, slx]*1
        self.patch_ll = ll_lores
        
        self.islx = slice(ll_lores[0], ll_lores[0]+2*self.patch_npix)
        self.isly = slice(ll_lores[1], ll_lores[1]+2*self.patch_npix)
        self.patch_shape = self.patch_seg[::self.pf,::self.pf].shape
    
        self.patch_sci = self.lores_im.data[self.isly, self.islx].flatten()
        self.patch_sivar = (self.lores_sivar[self.isly, self.islx]).flatten()
    
        self.patch_set_background(**bkg_kwargs)
        
        self.patch_wcs = self.lores_wcs.slice((self.islx, self.isly))
        self.patch_header = grizli.utils.get_wcs_slice_header(self.lores_wcs, 
                                              self.islx, self.isly)
        
        # Region mask
        mask_file = '{0}-{1}_mask.reg'.format(self.root, self.lores_filter)
        if os.path.exists(mask_file):
            import pyregion
            reg = pyregion.open('j055324m3342-ch1_mask.reg')
            
            grizli.utils.log_comment(self.LOGFILE, 
                   f'Use region mask: {mask_file}',
                   verbose=self.verbose, show_date=True)
            
            self.patch_reg_mask = ~reg.get_mask(header=self.patch_header, 
                                            shape=self.patch_shape).flatten()
        else:
            self.patch_reg_mask = np.isfinite(self.patch_sci)
            
        # Try to use stored patch transform
        model_image = '{0}-{1}_model.fits'.format(self.root, self.lores_filter)
        if os.path.exists(model_image):
            mh = pyfits.open(model_image)[0].header
            if 'RA_{0:03d}'.format(self.patch_id) in mh:
                rd = (mh['RA_{0:03d}'.format(self.patch_id)], 
                      mh['DEC_{0:03d}'.format(self.patch_id)])
                if np.array(self.patch_rd-np.array(rd)).sum() > 1.e-5:
                    self.patch_transform = np.array([0., 0., 0., 1.])
                else:
                    tf = [mh[f'DX_{self.patch_id:03d}'],
                          mh[f'DY_{self.patch_id:03d}'],
                          mh[f'ROT_{self.patch_id:03d}'],
                          mh[f'SCL_{self.patch_id:03d}']]
                    self.patch_transform = np.array(tf)
                    
                    msg = f'Use transform {tf} from {model_image}'
                    grizli.utils.log_comment(self.LOGFILE, msg,
                           verbose=self.verbose, show_date=True)

            else:
                self.patch_transform = np.array([0., 0., 0., 1.])
                
        else:
            self.patch_transform = np.array([0., 0., 0., 1.])
        
        # if ds9 is not None:
        #     ds9.frame(10)
        #     ds9.view(self.waterseg, header=grizli.utils.to_header(self.hst_wcs))
            
    @property 
    def patch_label(self):
        
        jname = grizli.utils.radec_to_targname(ra=self.patch_rd[0], dec=self.patch_rd[1], round_arcsec=(1./15, 1), precision=1, targstr='j{rah}{ram}{ras}.{rass}{sign}{ded}{dem}{des}.{dess}', header=None, )
        
        label = f'{self.root} {self.patch_id:03d} {self.lores_filter} {jname} {self.patch_npix:03d}'
        return label
        
    def patch_set_background(self, poly_order=-1, order_npix=64, order_clip=[3,11]):
        
        from numpy.polynomial.hermite import hermgrid2d as polygrid2d
        
        # Background model
        if poly_order < 0:
            auto_order = int(np.round(2*self.patch_npix/order_npix))
            poly_order = np.clip(auto_order, *order_clip)
        
        self.bkg_poly_order = poly_order
        
        x = np.linspace(-1, 1, 2*self.patch_npix)
        _Abg = []
        c = np.zeros((poly_order, poly_order))
        for i in range(poly_order):
            for j in range(poly_order):
                c *= 0
                c[i][j] = 1.e-3
                _Abg.append(polygrid2d(x, x, c).flatten())
        
        self._Abg = np.array(_Abg)
        self.Nbg = self._Abg.shape[0]
        
    def patch_compute_models(self, mag_limit=24, border_limit=0.1, use_saved_components=False, **kwargs):
        """
        Compute hst-to-IRAC components for objects in the patch
        """
        from golfir import irac
        
        hst_slice = self.hst_ujy[self.hst_sly, self.hst_slx]
        ids = np.unique(self.patch_seg[hst_slice != 0])[1:]

        if (self.phot['number'][ids-1] - ids).sum() == 0:
            mtest = self.phot['mag_auto'][ids-1] < mag_limit
            if mag_limit < 0:
                mtest = self.phot['flux_auto'][ids-1]/self.phot['fluxerr_auto'][ids-1] > -mag_limit
                msg = 'S/N > {0:.1f}'.format(-mag_limit, mtest.sum())
            else:
                msg = 'mag_auto < {0:.1f}'.format(mag_limit, mtest.sum())

            ids = ids[mtest]
        
        N = len(ids)
        self.patch_nobj = N
        self.patch_ids = ids
        
        grizli.utils.log_comment(self.LOGFILE, 
               f'Compute {N} models for patch ({msg})',
               verbose=self.verbose, show_date=True)
                
        # Shifts not used for now
        mx = my = None
        
        phot = self.phot
        
        # Translation
        patch_xy = self.lores_xy[self.patch_ids-1,:] - self.patch_ll
        
        xy_warp = irac.warp_catalog(self.patch_transform, patch_xy, 
                                    self.patch_sci.reshape(self.patch_shape),
                                    center=None)
        xy_offset = xy_warp - patch_xy
                    
        _A = []
        for i, id in tqdm(enumerate(ids)):
            #print(i, id)
            ix = phot['number'] == id
            
            if use_saved_components:
                _Ai = self.comp_to_patch(id)
                if _Ai is not None:
                    _A.append(_Ai)
                    continue
                    
            # if mx is not None:
            #     rx_i = mx(phot['ra'][ix]-m0[0], phot['dec'][ix]-m0[1])
            #     ry_i = my(phot['ra'][ix]-m0[0], phot['dec'][ix]-m0[1])
            #     tf = np.array([rx_i, ry_i]) #{'translation':(rx_i, ry_i)}
            # 
            # else:
            #     #tf = {'translation':(rx, ry)}
            #     tf = None
            
            _ = self.lores_psf_obj.evaluate_psf(ra=phot['ra'][ix][0], 
                                          dec=phot['dec'][ix][0], min_count=0, 
                                          clip_negative=True, 
                                          transform=xy_offset[i])

            lores_psf, psf_exptime, psf_count = _                              
            if (self.psf_window is -1) | (self.psf_only):
                psf_kernel = lores_psf
            else:
                #print('WINDOW {0}'.format(self.psf_window))
                psf_kernel = create_matching_kernel(self.hst_psf_full,
                                       lores_psf, window=self.psf_window)

            _Ai = convolve2d(hst_slice*(self.patch_seg == id), psf_kernel,
                              mode='constant', fft=1, cval=0.0)
                              
            _A.append(_Ai[self.pf//2::self.pf, self.pf//2::self.pf].flatten()*self.pf**2)                 

        self._A = np.array(_A)
        
        self.simple_model  = self._A.sum(axis=0).reshape(self.patch_shape)
        
        # Renormalize models so that coefficients are directly uJy
        self.Anorm = self._A.sum(axis=1)
        keep = self.Anorm > 0
        self._A = (self._A[keep,:].T/self.Anorm[keep]).T
        self.patch_ids = self.patch_ids[keep]
        self.Anorm = self.Anorm[keep]
        self.patch_nobj = keep.sum()
        self.model_bright = 0.
        
        # Patch border
        nx = self.patch_npix*self.pf
        border = np.ones((2*nx, 2*nx))
        yp, xp = np.indices((border.shape))
        R = np.sqrt((xp+self.pf-nx)**2+(yp+self.pf-nx)**2)
        border[1:-1,1:-1] = 0
        #border = (R > Npan*np)*1
        b_conv = convolve2d(border, psf_kernel, mode='constant', fft=1,
                            cval=1)
                                     
        self.patch_border = b_conv[::self.pf, ::self.pf].flatten()
        self.patch_border /= self.patch_border.max()
        self.patch_border_mask = self.patch_border < border_limit
        self.patch_border_limit = border_limit
        
        # IDs that touch the border
        self.patch_seg_sub = self.patch_seg[self.pf//2::self.pf, self.pf//2::self.pf].flatten()
        
        # self.patch_border_ids = np.unique(self.patch_seg_sub*(~self.patch_border_mask))[1:]
        # self.patch_ids_in_border = grizli.utils.column_values_in_list(self.patch_ids, self.patch_border_ids)

        # IDs that fall out of the (bordered) patch
        
        patch_xyint = np.clip(np.cast[int](np.round(patch_xy)), 0, self.patch_shape[-1]-1)
        self.patch_ids_in_border = ~self.patch_border_mask.reshape(self.patch_shape)[patch_xyint[:,1], patch_xyint[:,0]]
        self.patch_border_ids = self.patch_ids[self.patch_ids_in_border]
        
        # Bright limits
        self.patch_bright_limits(**kwargs)
    
    def patch_iraclean(self):
        
        yp, xp = np.indices(self.patch_shape)
        
        phot = self.phot
        
        _Af = np.vstack([self._Abg, self._A])  
        
        mask = self.patch_sivar > 0
        #mask &= self.model_bright*self.patch_sivar < self.bright_args['bright_sn']
        mask &= self.patch_border_mask
        mask &= self.patch_reg_mask
        
        patch_mask = mask.reshape(self.patch_shape)
        
        ids = self.patch_ids
        
        for ie, id_i in enumerate(ids):
            in_seg = (self.patch_seg_sub == id_i).reshape(self.patch_shape)
            if in_seg.sum() == 0:
                continue
            
            ix = phot['number'] == id_i
            
            _ = self.lores_psf_obj.evaluate_psf(ra=phot['ra'][ix][0], 
                                  dec=phot['dec'][ix][0], 
                                  min_count=1, clip_negative=True, 
                                  transform=None)
                                  
            psf_full_i, _exptime, _count = _    
            # Resample
            psf_i = psf_full_i[self.pf//2::self.pf, self.pf//2::self.pf]
            psf_i /= psf_i.sum()
            sh = psf_i.shape
            sx = sh[1]//2
            
            edge = (xp[in_seg].min()-sx < 0) 
            edge |= (xp[in_seg].max()+sx > self.patch_shape[1])
            edge |= (yp[in_seg].min()-sx < 0) 
            edge |= (yp[in_seg].max()+sx > self.patch_shape[0])
            
            if edge:
                continue
            
            psfobj = np.zeros((in_seg.sum()+1, self.patch_shape[0], self.patch_shape[1]))
            psfobj[0,:,:] = 1.
            
            for i, (xi, yi) in enumerate(zip(xp[in_seg], yp[in_seg])):
                slx = slice(xi-sx,xi-sx+sh[1])
                sly = slice(yi-sx,yi-sx+sh[0])
                psfobj[i+1, sly, slx] += psf_i
            
            _Ap = psfobj.reshape((in_seg.sum()+1, -1))
            _Asum = _Ap[1:,:].sum(axis=0)
            xmsk = patch_mask.flatten() & (_Asum > 0)
            xmsk &= _Asum > 0.05*_Asum.max()
            
            # Remove object
            _c = self.patch_coeffs*1
            ii = np.where(self.patch_ids == id_i)[0][0]
            _c[self.Nbg + ii] = 0

            h_model = _Af.T.dot(_c)
            ydata = self.patch_sci - h_model
            xmsk &= (ydata*self.patch_sivar > -1)
            
            _Axp = (_Ap*self.patch_sivar).T[xmsk]
            _yp = (ydata*self.patch_sivar)[xmsk]    
            
            # iraclean model
            include_background = False
            
            if include_background:
                _xp = np.linalg.lstsq(_Axp, _yp, rcond=-1)
                pbg = _xp[0][0]
                p_model = _Ap.T.dot(_xp[0])
                msum = _xp[0][1:].sum()
            else:
                _xp = np.linalg.lstsq(_Axp[:,1:], _yp, rcond=-1)
                pbg = 0.
                p_model = _Ap[1:].T.dot(_xp[0])
                msum = _xp[0].sum()
   
    def comp_to_patch(self, id, full_image=False, flatten=True):
        """
        Put a stored component into the patch coordinates
        """    
        ext = ('MODEL',id)
        if ext not in self.comp_hdu:
            return None
        
        hdu = self.comp_hdu[ext]
        if full_image:
            patch = np.zeros(self.lores_shape, dtype=np.float32)
            ll = [0,0]
        else:
            patch = np.zeros(self.patch_shape, dtype=np.float32)
            ll = self.patch_ll
        
        slx = slice(hdu.header['XMIN']-ll[0], hdu.header['XMAX']-ll[0])
        sly = slice(hdu.header['YMIN']-ll[1], hdu.header['YMAX']-ll[1])
        try:
            patch[sly, slx] += hdu.data
            if flatten:
                return patch.flatten()
            else:
                return patch
                
        except:
            print(f'comp_to_patch({id}) failed')
            return None
            
    def patch_bright_limits(self, any_limit=12, point_limit=16, point_flux_radius=3.5, bright_ids=None, bright_sn=7, **kwargs):
        """
        Set masking for bright objects
        """
        phot = self.phot
        
        bright = (phot['mag_auto'][self.patch_ids-1] < any_limit) 
        bright |= ((phot['mag_auto'][self.patch_ids-1] < point_limit) &
                  (phot['flux_radius'][self.patch_ids-1] < point_flux_radius))
                
        if bright_ids is not None:
            for ii in bright_ids:
                bright[self.patch_ids == ii] = True
        
        self.patch_is_bright = bright
        
        self.bright_args = dict(any_limit=any_limit, point_limit=point_limit, point_flux_radius=point_flux_radius, bright_ids=bright_ids, bright_sn=bright_sn)
        msg = f'Set bright target mask: {self.bright_args}'
        grizli.utils.log_comment(self.LOGFILE, msg, verbose=self.verbose, 
               show_date=True)
         
        self.model_bright = self._A[bright,:].T.dot(self.Anorm[bright])
        
    def patch_least_squares(self):
        """
        Least squares coeffs
        """
        
        msg = 'Patch least squares'
        grizli.utils.log_comment(self.LOGFILE, msg, verbose=self.verbose, 
               show_date=True)
        
        _Af = np.vstack([self._Abg, self._A])  
        
        y = self.patch_sci #self.lores_im.data[self.isly, self.islx].flatten()
        
        mask = self.patch_sivar > 0
        mask &= self.model_bright*self.patch_sivar < self.bright_args['bright_sn']
        mask &= self.patch_border_mask
        mask &= self.patch_reg_mask
        
        self.patch_mask = mask.reshape(self.patch_shape)
        
        self._Ax = (_Af[:,mask]*self.patch_sivar[mask]).T
        _yx = (y*self.patch_sivar)[mask]
        _x = np.linalg.lstsq(self._Ax, _yx, rcond=-1)
        
        self.patch_model = _Af.T.dot(_x[0]).reshape(self.patch_shape)
        
        self.patch_resid = self.patch_sci.reshape(self.patch_shape) - self.patch_model
        
        _xbg = _x[0][:self.Nbg]
        self.patch_bg = _Af[:self.Nbg].T.dot(_xbg).reshape(self.patch_shape)
        
        self.patch_lstsq = _x
        self.patch_coeffs = _x[0]*1
        self.patch_bkg_coeffs = _x[0][:self.Nbg]*1
        self.patch_obj_coeffs = _x[0][self.Nbg:]*1
        return _x
        
    def compute_err_scale(self, max_sn=1):
        """
        Rescale uncertainties based on residuals
        """    
        
        ##############
        # Compute scaling of error array based on residuals
        err_resid = self.patch_resid.flatten()*self.patch_sivar

        resid_mask = self.patch_mask.flatten() 
        resid_mask &= (self.patch_model.flatten()*self.patch_sivar < max_sn)
        
        ERR_SCALE_i = grizli.utils.nmad(err_resid[resid_mask]) 
        msg = f'patch {self.patch_id:03d}: ERR_SCALE_i =  {ERR_SCALE_i:.3f}'
        grizli.utils.log_comment(self.LOGFILE, msg, verbose=self.verbose, 
               show_date=True)
        
        self.ERR_SCALE *= ERR_SCALE_i
        
        # Update patch_sivaar (property lores_sivar includes ERR_SCALE)
        self.patch_sivar = (self.lores_sivar[self.isly, self.islx]).flatten()
                
    def patch_display(self, ds9=None, figsize=[10,10], subplots=220, savefig='png', cmap='Spectral_r', vm=(-0.03, 0.4), horizontal=False):
        
        if ds9 is not None:
            ds9.frame(12)
            ds9.view(self.lores_im.data, header=self.lores_im.header)

            ds9.frame(13)
            ds9.view(self.patch_model*self.patch_mask,
                     header=self.patch_header)

            ds9.frame(14)
            ds9.view(self.patch_resid*self.patch_mask,
                     header=self.patch_header)
            
            ds9.frame(15)
            ds9.view(self.patch_resid*self.patch_mask,
                     header=self.patch_header)
        else:
            # Figure

            if savefig is not None:
                plt.ioff()
                
            if horizontal:
                figsize=[8, 2.5]
                subplots=140
                
            fig = plt.figure(figsize=figsize)

            ax = fig.add_subplot(subplots+1)
            hst_slice = self.hst_ujy[self.hst_sly, self.hst_slx]
            ax.imshow(hst_slice, vmin=vm[0]/10, vmax=vm[1]/10, cmap=cmap)
            ax.set_xlabel('HST {0}'.format(self.ref_filter.upper()))

            ax = fig.add_subplot(subplots+2)
            ax.imshow(self.patch_sci.reshape(self.patch_shape), vmin=vm[0], vmax=vm[1], cmap=cmap)
            ax.set_xlabel('{0}'.format(self.lores_filter))

            ax = fig.add_subplot(subplots+3)
            ax.imshow(self.patch_model*self.patch_mask, vmin=vm[0], vmax=vm[1], cmap=cmap)
            ax.set_xlabel('{0} Model'.format(self.lores_filter))
                                
            ax = fig.add_subplot(subplots+4)
            ax.imshow(self.patch_resid*self.patch_mask, vmin=vm[0], vmax=vm[1], cmap=cmap)
            ax.set_xlabel('{0} Residual'.format(self.lores_filter))

            for ax in fig.axes:
                ax.set_xticklabels([])
                ax.set_yticklabels([])
            
            if horizontal:
                ax_i = fig.axes[2]
            else:
                ax_i = fig.axes[1]

            # Global title
            fig.text(0.5, 0.97, self.patch_label, ha='center', va='top', 
                    transform=fig.transFigure, fontsize=10)
                
            fig.tight_layout(pad=0.5)
            
            if savefig is not None:
                fig.savefig('{0}.{1}'.format(self.patch_label.replace(' ', '_'), savefig))
                plt.ion()
                plt.close('all')
                
    def patch_align(self):
        """
        Realign
        """
        from golfir.utils import get_wcslist, _obj_shift
        from golfir import irac
        
        t0 = np.array([0,0,0,1])*np.array([1,1,1,100.])

        args = (self.lores_im.data[self.isly, self.islx], self.patch_model, self.patch_sivar.reshape(self.patch_shape)*self.patch_mask, 0)

        _res = minimize(_obj_shift, t0, args=args, method='powell')

        args = (self.lores_im.data[self.isly, self.islx], self.patch_model, self.patch_sivar.reshape(self.patch_shape)*self.patch_mask, 1)
        tfx, warped = _obj_shift(_res.x, *args)
        tf = tfx*1

        # Modify PSF positioning
        dd = np.cast[int](np.round(tf[:2]*self.pf))
        self.patch_transform[:2] += tf[:2]*self.pf
        self.patch_transform[2] += tf[2]
        self.patch_transform[3] *= tf[3]

        #self.patch_transform[3] *= tf[3]
        
        msg = 'Patch {0:03d}, transform: ({1:.2f}, {2:.2f}) {3:6.3f} {4:.3f}'.format(self.patch_id, *self.patch_transform)
        grizli.utils.log_comment(self.LOGFILE, msg, verbose=self.verbose, 
               show_date=True)
        
        # hst_psf_offset -= dd
        # 
        # ds9.frame(16)
        # ds9.view(msk2d*(irac_im.data[isly, islx]-warped), header=utils.get_wcs_slice_header(irac_wcs, islx, isly))   

        for i in range(self.patch_nobj):
            self._A[i,:] = irac.warp_image(tf, self._A[i,:].reshape(self.patch_shape)).flatten()
        
        # Reset bright mask
        self.patch_bright_limits(**self.bright_args)
        
        self.patch_least_squares()
    
    # Sersic -> Gaussian 
    GAUSS_KEYS = {'n':0.5, 'R_e':0.05, 'q':1., 'pa':0., 
                  'fix':{'n':0, 'R_e':1, 'q':0, 'pa':0}}

    def patch_fit_galfit(self, ids=[], dil_size=11, component_type=None, match_geometry=False, fit_sky=True, component_keys={}, min_err=0.00, chi2_threshold=10, ds9=None, use_oversampled_psf=False, use_flat_rms=True, **kwargs):
        """
        Fit individual sources with Galfit
        
        If an element of `ids` is negative, fit all sources whose segments 
        touch that object.
        
        """
        from grizli.galfit import galfit
        from grizli.galfit.galfit import GalfitPSF, GalfitSersic, GalfitSky
        from grizli.galfit.galfit import Galfitter as _fitter
        
        fit_ids = []
        
        for id in ids:            
            if id < 0:
                # IDs that touch
                xmsk = (seg_sl == -id)
                msk_dil = binary_dilation(xmsk, np.ones((dil_size,dil_size)))
                fit_ids.extend(list(np.unique(seg_sl[msk_dil])[1:]))
            else:
                fit_ids.append(id)

        patch_seg = self.patch_seg[self.pf//2::self.pf, self.pf//2::self.pf]
        segmap = (patch_seg == 0)

        segmap &= (self.patch_sivar > 0).reshape(self.patch_shape)
        segmap &= self.patch_border_mask.reshape(self.patch_shape)

        _c = self.patch_coeffs*1.
        _c0 = _c*1.

        _Af = np.vstack([self._Abg, self._A])  
        patch_model = _Af.T.dot(_c0).reshape(self.patch_shape)

        pop_ids = []

        initial_fluxes = self.patch_obj_coeffs*1

        galfit_psf = None
        
        # Copy
        phot = self.phot

        # Sky component
        if fit_sky:
            components = [GalfitSky()]
        else:
            components = []
        
        # Object components
        for i, id_i in enumerate(fit_ids):
            
            # Unmask object segments
            ix = phot['number'] == id_i
            s_ix = np.arange(self.patch_nobj)[self.patch_ids == id_i]
            _c[self.Nbg + s_ix] = 0
            segmap |= (patch_seg == id_i)
            
            # Galfit Components
            if component_type in ['psf']:
                components += [galfit.GalfitPSF()]
            elif component_type in ['gaussian']:
                gauss = galfit.GalfitSersic(**self.GAUSS_KEYS)
                components += [gauss]
            elif hasattr(component_type, 'mro'):
                components += [component_type(**component_keys)]
                print('Use component {0} for {1}'.format(component_type.name, id_i))
            elif hasattr(component_type, '__len__'):
                print('Use component {0} ({1}) for {2}'.format(component_type[i].name, component_type[i].pdict, id_i))
                components += [component_type[i]]
            else:
                keys = {'q':0.8}
                if match_geometry:
                    ix = phot['number'] == id_i
                    q = (phot['b_image']/phot['a_image']).data[ix][0]
                    pa = phot['theta_image'].data[ix][0]/np.pi*180+90
                    keys['pa'] = pa
                    keys['q'] = q
                    print('Fix parameters {0} for id={1}'.format(keys, id_i))
                    keys['fix'] = {'q':0, 'pa':0}

                components += [galfit.GalfitSersic(**keys)]#, R_e=0.5)]
            
            # Evaluate PSF at location of object
            if galfit_psf is None:
                _ = self.lores_psf_obj.evaluate_psf(ra=phot['ra'][ix][0], 
                                      dec=phot['dec'][ix][0], 
                                      min_count=1, clip_negative=True, 
                                      transform=None)
                                      
                galfit_psf_full, galfit_psf_exptime, galfit_psf_count = _    
                
                # Resample
                if not use_oversampled_psf:
                    galfit_psf = galfit_psf_full[self.pf//2::self.pf, self.pf//2::self.pf]
                else:
                    galfit_psf = galfit_psf_full*1
                    
            xy = self.lores_xy[ix,:][0] - self.patch_ll + 0.5
            components[-1].pdict['pos'] = list(xy)
            components[-1].pdict['mag'] = 22.5

            if (id_i in self.patch_ids) & True:
                fclip = np.maximum(initial_fluxes[self.patch_ids == id_i], 1)
                components[-1].pdict['mag'] = 26.563-2.5*np.log10(fclip)[0]
                if components[-1].pdict['mag'] > 28:
                    pop_ids.append(id_i)
                    components.pop(-1)
                    continue

        for id in pop_ids:
            fit_ids.pop(fit_ids.index(id))

        # Model without sources to fit
        sci_cleaned = (self.patch_sci - _Af.T.dot(_c)).reshape(self.patch_shape)

        if ds9:
            ds9.frame(15)
            ds9.view(sci_cleaned*segmap, header=self.patch_header)

        ivar = self.patch_sivar.reshape(self.patch_shape)**2
        if use_flat_rms:
            ivar = ivar*0.+np.median(ivar[self.patch_mask])
            
        # Min error
        ivar = 1/(1/ivar+(min_err*sci_cleaned)**2)
        
        # Full initial model
        chi2_init = (self.patch_resid**2*ivar)[segmap].sum()
        dof = segmap.sum()

        if ds9:
            ds9.frame(13)
            ds9.view(self.patch_model*segmap, header=self.patch_header)

            ds9.frame(14)
            ds9.view(self.patch_resid*segmap, header=self.patch_header)

        self.galfit_psf = galfit_psf
        self.galfit_ivar = ivar
        self.galfit_segmap = segmap
        self.galfit_cleaned = sci_cleaned
        
        self.galfit_fit_ids = fit_ids
        self.galfit_components = components
        
        self.galfit_chi2_init = chi2_init
        self.galfit_dof = dof
        
        try:
            
            if use_oversampled_psf:
                psf_sample = self.pf
            else:
                psf_sample = 1
                
            gf_root = f'gf{self.patch_id}{self.lores_filter}'
            gf = _fitter.fit_arrays(sci_cleaned, ivar*segmap, segmap*1, 
                                galfit_psf, psf_sample=psf_sample, id=1, 
                                components=components, recenter=False, 
                                exptime=0, path=os.getcwd(), root=gf_root)
            
            chi2_final = (gf['resid'].data**2*ivar)[segmap].sum()

            if ds9:
                ds9.frame(15)
                ds9.view(gf['resid'].data*segmap, header=self.patch_header)
            
        except:
            grizli.utils.log_exception('galfit.failed', traceback, verbose=True)
            chi2_final = 1e30
            gf = None
            
        self.galfit_result = gf
        
        self.galfit_chi2_final = chi2_final
        chi2_diff = chi2_init - chi2_final
        msg = (f'patch_galfit:    fit_ids = {fit_ids}\n'
               f'patch_galfit:  chi2_init = {chi2_init/dof:.2f}\n'
               f'patch_galfit: chi2_final = {chi2_final/dof:.2f}\n'
               f'patch_galfit:       diff = {chi2_diff:.1f}')
                    
        grizli.utils.log_comment(self.LOGFILE, msg, verbose=self.verbose, 
               show_date=True)
        
        if chi2_diff  > chi2_threshold:
            self.insert_galfit_models(ds9=ds9)
            
    def insert_galfit_models(self, ds9=None):
        
        # Component imaage
        cwd = os.getcwd()
        #os.chdir('/tmp/')
        gf_root = f'gf{self.patch_id}{self.lores_filter}'
        
        os.system(f'perl -pi -e "s/P\) 0/P\) 3/" {gf_root}.gfmodel')
        os.system(f'galfit {gf_root}.gfmodel')
        im = pyfits.open('subcomps.fits')
        
        #os.chdir(cwd)

        for i, id in enumerate(self.galfit_fit_ids):
            gf_model = im[i+2].data.flatten()
            self._A[self.patch_ids == id,:] = gf_model/gf_model.sum()
            
            j = i+2
            pars = {'id':id, 'type':im[1].header[f'COMP_{j}'], 'll':self.patch_ll}
            
            trans = {'ar':'q'}
            
            for k in im[1].header:
                if k.startswith(f'{j}_'):
                    val = im[1].header[k]
                    star = ('*' in val)*1
                    kspl = k.split('_')[1].lower()
                    if kspl in trans:
                        kspl = trans[kspl]
                        
                    pars[kspl] = (float(val.replace('*','').split()[0]), star)
            
            msg = f'insert_galfit_models: {pars}'        
            grizli.utils.log_comment(self.LOGFILE, msg, verbose=self.verbose, 
                   show_date=True)
            
            self.galfit_parameters[id] = pars
        
        self.patch_least_squares()
        
        if ds9:
            ds9.frame(15)
            ds9.view(self.patch_resid*self.patch_mask,
                     header=self.patch_header)
    
    def save_patch_results(self, min_clip=-2, clip_border=True, multicomponent_file=True):
        """
        Save results to the model image and photometry file
        """
        
        ##############
        # Image
        model_image = '{0}-{1}_model.fits'.format(self.root, self.lores_filter)
        if os.path.exists(model_image):
            im_model = pyfits.open(model_image)[0]
            self.full_model = im_model.data
            h = im_model.header
        else:
            self.full_model = self.lores_im.data*0
            h = self.lores_im.header

        self.full_model[self.isly, self.islx][self.patch_mask] = self.patch_model[self.patch_mask]
        
        # Update patch info in header
        pk = '{0:03d}'.format(self.patch_id)
        h['RA_'+pk] = (self.patch_rd[0], 'Patch RA')
        h['DEC_'+pk] = (self.patch_rd[1], 'Patch DEC')
        h['ARCM_'+pk] = (self.patch_npix, 'Patch size arcmin')
        h['NPIX_'+pk] = (self.patch_npix, 'Patch size pix')
        h['DX_'+pk] = (self.patch_transform[0], 'Patch x shift')
        h['DY_'+pk] = (self.patch_transform[1], 'Patch y shift')
        h['ROT_'+pk] = (self.patch_transform[2], 'Patch rot')
        h['SCL_'+pk] = (self.patch_transform[3], 'Patch scale')
        
        pyfits.writeto(model_image, data=self.full_model, header=h, 
                       overwrite=True)
        
        ################
        # Fluxes and uncertainties from the least squares fit
        flux = self.patch_obj_coeffs*1
        covar = np.matrix(np.dot(self._Ax.T, self._Ax)).I.A
        err = np.sqrt(covar.diagonal())[self.Nbg:]

        # Clip very negative measurements
        bad = flux < min_clip*err
        flux[bad] = -99.*2
        err[bad] = -99.*2
        
        phot = self.phot
        
        # Put ERR_SCALE in catalog header
        phot.meta['{0}_ERR_SCALE'.format(self.column_root.upper())] = self.ERR_SCALE

        ix = self.patch_ids - 1
        
        if clip_border:            
            ix = ix[~self.patch_ids_in_border]
            flux = flux[~self.patch_ids_in_border]
            err = err[~self.patch_ids_in_border]
            
        column_root = self.column_root

        phot['{0}_flux'.format(column_root)][ix] = flux
        phot['{0}_err'.format(column_root)][ix] = err
        phot['{0}_patch'.format(column_root)][ix] = self.patch_id
        
        if 'mips' in self.column_root:
            apcorr = self.lores_psf_obj.apcorr
            phot['{0}_flux'.format(column_root)][ix] *= apcorr
            phot['{0}_err'.format(column_root)][ix] *= apcorr
            
            phot['F325'] = phot['mips_24_flux']
            phot['E325'] = phot['mips_24_err']
            phot.meta['{0}_apcorr'] = apcorr
                    
        if self.patch_is_bright is not None:
            ix_bright = (self.patch_ids - 1)[self.patch_is_bright]
            phot['{0}_bright'.format(column_root)][ix_bright] = 1

        phot.write(f'{self.root}_irac_phot.fits', overwrite=True)
        
        if multicomponent_file:
            self.save_multicomponent_model()
            
    def save_multicomponent_model(self, skip_existing=False):
        """
        Model image with fully separate component layers
        """
        component_file = f'{self.root}-{self.lores_filter}_components.fits'
        # if os.path.exists(comp_file):
        #     comp_hdu = pyfits.open(comp_file)
        #     full_mask = comp_hdu[0].data
        # else:
        #     full_mask = np.zeros(self.lores_im.data.shape, dtype=np.uint8)
        #     comp_hdu = pyfits.HDUList([pyfits.PrimaryHDU(header=self.lores_im.header, data=full_mask)])
        #     comp_hdu[0].header['EXTNAME'] = 'MASK'
            
        isly, islx = self.isly, self.islx
        self.full_mask[isly, islx] |= self.patch_mask

        wsl = self.patch_wcs
        
        # Slices for subcomponent models
        yp, xp = np.indices(self.patch_shape)
        flux = self.patch_obj_coeffs*1

        for i, id in tqdm(enumerate(self.patch_ids)):
            extn = 'MODEL',id
            if (extn in self.comp_hdu):
                if skip_existing:
                    print(f'Extension ({extn}) exists, skip.')
                    continue
                else:
                    self.comp_hdu.pop(extn)
                
            ix = self.patch_ids == id
            _Ai = self._A[ix,:].reshape(self.patch_shape)

            nonz = _Ai/_Ai.max() > 1.e-6

            xpm = xp[nonz]
            ypm = yp[nonz]

            # slice in patch
            sislx = slice(xpm.min(), xpm.max())
            sisly = slice(ypm.min(), ypm.max())
            
            # global slice
            fislx = slice(sislx.start + islx.start, sislx.stop + islx.start)
            fisly = slice(sisly.start + isly.start, sisly.stop + isly.start)

            sub_wcs = self.lores_wcs.slice((fisly, fislx))
            sub_head = grizli.utils.to_header(sub_wcs)

            hdu_i = pyfits.ImageHDU(data=(_Ai*nonz*flux[i])[sisly, sislx],
                                    header=sub_head)

            hdu_i.header['ID'] = (id, 'Object ID number')
            hdu_i.header['EXTNAME'] = 'MODEL'
            hdu_i.header['EXTVER'] = id
            hdu_i.header['NPIX'] = nonz.sum()

            hdu_i.header['FLUX_UJY'] = (flux[i], 'Total flux density, uJy')
            #hdu_i.header['ERR_UJY'] = (err[i], 'Total flux uncertainty, uJy')

            hdu_i.header['XMIN'] = (sislx.start + islx.start, 'x slice min')
            hdu_i.header['XMAX'] = (sislx.stop + islx.start, 'x slice max')
            hdu_i.header['YMIN'] = (sisly.start + isly.start, 'y slice min')
            hdu_i.header['YMAX'] = (sisly.stop + isly.start, 'y slice max')
            
            hdu_i.header['GALFIT'] = (id in self.galfit_parameters, 
                                      'Model is from Galfit')
            if hdu_i.header['GALFIT']:
                print('Galfit component: {0}'.format(id))
                params = self.galfit_parameters[id]
                for p in params:
                    if p in 'id':
                        continue
                    
                    if p in ['ll']:
                        for j in range(len(params[p])):
                            pk = 'GF_{0}_{1}'.format(p, j)
                            hdu_i.header[pk] = params[p][j]
                    elif p in ['type']:
                        pk = 'GF_{0}'.format(p)
                        hdu_i.header[pk] = (params[p], 
                                            'Galfit component type')
                    else:
                        pk = 'GF_{0}'.format(p)
                        hdu_i.header[pk] = (params[p][0], f'Galfit param {p}')
                        pk = 'GF_{0}_F'.format(p)
                        hdu_i.header[pk] = (params[p][1], 'Starred parameter')
                        
            if ('MODEL',id) in self.comp_hdu:
                self.comp_hdu[extn] = hdu_i
            else:
                self.comp_hdu.append(hdu_i)

        self.comp_hdu.writeto(component_file, overwrite=True)
                

    def run_full_patch(self, rd_patch=None, patch_arcmin=1.4, ds9=None, patch_id=0, mag_limit=[24,27], galfit_flux_limit=None, match_geometry=False, refine_brightest=True, bkg_kwargs={}, run_alignment=True, galfit_kwargs={}, rescale_uncertainties=False, **kwargs):
        """
        Run the multi-step process on a single patch
        """
        
        if not hasattr(mag_limit, '__len__'):
            #mag_limit = [mag_limit, mag_limit]
            run_alignment=False
            
        if False:
            self.run_full_patch(rd_patch=None, patch_arcmin=patch_arcmin, ds9=ds9, patch_id=patch_id, mag_limit=mag_limit, galfit_flux_limit=10, match_geometry=False)
            
        # Initialize patch, make cutouts, etc.
        self.patch_initialize(rd_patch=rd_patch, patch_arcmin=patch_arcmin, ds9=ds9, patch_id=patch_id, bkg_kwargs=bkg_kwargs)
        
        # First pass on models
        self.patch_compute_models(mag_limit=mag_limit[0], **kwargs)
        self.patch_least_squares()
                
        # Alignment
        if run_alignment:
            self.patch_align()
        
            if ds9:
                self.patch_display(ds9=ds9)
        
            # Regenerate models with alignment transform
            self.patch_compute_models(mag_limit=mag_limit[1], **kwargs)
            self.patch_least_squares()
        
        # Error scaling
        if rescale_uncertainties:
            self.compute_err_scale()  
        
        if ds9:
            ds9.frame(15)
            ds9.view(self.patch_resid*self.patch_mask,
                 header=self.patch_header)
                                  
        # Galfit refinement
        if galfit_flux_limit is not None:
            # bright ids
            ix = self.patch_ids -1
            
            # flux = self.patch_obj_coeffs*1
            # covar = np.matrix(np.dot(self._Ax.T, self._Ax)).I.A
            # err = np.sqrt(covar.diagonal())[self.Nbg:]
            # 
            fit_bright = self.patch_obj_coeffs > galfit_flux_limit
            #fit_bright &= (~self.patch_ids_in_border)
            fit_bright &= (~self.patch_is_bright)
            
            so = np.argsort(self.patch_obj_coeffs[fit_bright])[::-1]
            bright_ids = self.patch_ids[fit_bright][so]
            #fit_with_psf = self.phot['flux_radius'][ix][fit_bright][so] < 1.5
            #fit_with_psf=False
            fit_with_psf = self.phot['flux_radius'][ix][fit_bright][so] < 0
            
            for id, is_psf in zip(bright_ids, fit_with_psf):
                if is_psf:
                    comp = 'psf'
                else:
                    comp = None
                
                self.patch_fit_galfit(ids=[id], component_type=comp,
                                      chi2_threshold=10, 
                                      match_geometry=match_geometry, 
                                      **galfit_kwargs)
            
                if ds9:
                    ds9.frame(18)
                    ds9.view(self.patch_resid*self.patch_mask,
                             header=self.patch_header)
                   
        # Refine error scaling
        if rescale_uncertainties:
            self.compute_err_scale()  
        
        # Refine galfit for brightest
        if refine_brightest:
            bright_ids = -1
            bright_ids = list(self.patch_ids[self.patch_is_bright])
            
            self.bright_args['any_limit'] = 10
            self.bright_args['point_limit'] = 10
            
            self.patch_bright_limits(**self.bright_args)
            for id in bright_ids:
                for comp in ['psf',None]:
                    self.patch_fit_galfit(ids=[id], component_type=comp,
                                          chi2_threshold=10, 
                                          match_geometry=match_geometry)
                
                if id in self.galfit_parameters:
                    if id in bright_ids:
                        bright_ids.pop(bright_ids.index(id))
                    
                    if ds9:
                        ds9.frame(18)
                        ds9.view(self.patch_resid*self.patch_mask,
                                 header=self.patch_header)
            
            self.bright_args['bright_ids'] = bright_ids
            self.patch_bright_limits(**self.bright_args)

            self.patch_least_squares()
            # Error scaling
            if rescale_uncertainties:
                self.compute_err_scale()  
            
            if ds9:
                ds9.frame(18)
                ds9.view(self.patch_resid*self.patch_mask,
                         header=self.patch_header)
                                
            
        # Save diagnostic figure
        self.patch_display(ds9=None)
        
        # Save photometry & global model
        self.save_patch_results()
    
    def generate_patches(self, patch_arcmin=1.0, patch_overlap=0.2, **kwargs):
        """
        Generate patches to tile the field
        """
        
        size =  patch_arcmin*60/0.5
        pad = 1.02*size
        extra = 10./0.5 # extra padding, arcsec
        step = (2*patch_arcmin-patch_overlap)*60/0.5
        
        valid = np.isfinite(self.phot['mag_auto'])
        
        if patch_arcmin < 0:
            size = 0.
            pad = 10
            
        xmin = np.maximum(self.lores_xy[valid,0].min()+size-extra, pad)
        xmax = np.minimum(self.lores_xy[valid,0].max(),  
                          self.lores_shape[1]-pad)
        
        ymin = np.maximum(self.lores_xy[valid,1].min()+size-extra, pad)
        ymax = np.minimum(self.lores_xy[valid,1].max(), 
                          self.lores_shape[0]-pad)
        
        if patch_arcmin < 0:
            # Single patch
            xp = np.array([(xmax + xmin)/2.])
            yp = np.array([(ymax + ymin)/2.])
            
            rp, dp = self.lores_wcs.all_pix2world(xp[:1].flatten(), yp[:1].flatten(), 0)
            
            cosd = np.cos(dp[0]/180*np.pi)
            patch_arcmin = np.maximum((xmax-xmin)*cosd*60, (ymax-ymin)*60)
            
        else:    
            xp, yp = np.meshgrid(np.arange(xmin, xmax+size, step), np.arange(ymin, ymax+size, step))
        
            rp, dp = self.lores_wcs.all_pix2world(xp.flatten(), yp.flatten(), 0)
        
        patch_ids = np.arange(len(rp))
        
        fp = open(f'{self.root}_patch.reg','w')
        fp.write('fk5\n')
        for i in range(len(rp)):
            fp.write(f'box({rp[i]:.6f}, {dp[i]:.6f}, '
                     f'{2*patch_arcmin}\', {2*patch_arcmin}\') # text=xxx{patch_ids[i]}yyy\n'.replace('xxx','{').replace('yyy','}'))
        fp.close()
        
        tab = grizli.utils.GTable()
        tab['patch_id'] = patch_ids
        tab['ra'] = rp
        tab['dec'] = dp
        tab['patch_arcmin'] = patch_arcmin
        
        tab.write(f'{self.root}_patch.fits', overwrite=True)
        return tab

RUN_ALL_DEFAULTS = {'ds9':None, 'patch_arcmin':1., 'patch_overlap':0.2, 'mag_limit':[24,27], 'galfit_flux_limit':20, 'refine_brightest':True, 'run_alignment':True, 'any_limit':18, 'point_limit':17, 'bright_sn':10, 'bkg_kwargs':{'order_npix':64}, 'channels':['ch1','ch2','ch3','ch4'], 'psf_only':False, 'use_saved_components':True, 'window':None} 
      
def run_all_patches(root, PATH='/GrizliImaging/', ds9=None, sync_results=True, channels=['ch1','ch2','ch3','ch4'], fetch=True, use_patches=True, **kwargs):
    """
    Generate and run all patches
    """
    import os
    import glob
    import time
    
    import grizli.utils
    import golfir.model
    
    if True:
        try:
            os.chdir(PATH)
        except:
            os.chdir('/Users/gbrammer/Research/HST/CHArGE/IRAC/')
            
        if not os.path.exists(root):
            os.mkdir(root)
        
        os.chdir(root)
        if fetch & (not os.path.exists(f'{root}-ch1_drz_sci.fits')):
            golfir.model.ImageModeler.fetch_from_aws(root)
        
    try:
        _ = ds9
    except:
        ds9 = None
    
    if False:
        scl = {'ch1':1.02, 'ch2':1.02, 'ch3':1.0, 'ch4':1.0}
        for ch in scl:
            psf_file = glob.glob(f'../AvgPSF/irac_{ch}_*[md]_psf.fits')[-1]
            print(psf_file)
            psf = pyfits.open(psf_file)
            psf[0].data = psf[0].data**(scl[ch])
            psf.writeto(f'{root}-{ch}-0.1.psfr_avg.fits', overwrite=True)
            
    if ds9:
        PWD = os.getcwd()
        ds9.frame(1)
        ds9.set(f'file {PWD}/{root}-ir_drz_sci.fits')
        ds9.set_defaults(match='wcs')
        ds9.frame(2)
        ds9.set(f'file {PWD}/{root}-ir_seg.fits')
        ds9.frame(3)
        ds9.set(f'file {PWD}/{root}-ch1_drz_sci.fits')
        ds9.frame(4)
        ds9.set(f'file {PWD}/{root}-ch2_drz_sci.fits')

        ds9.set('frame lock wcs')
        ds9.set('lock colorbar')

        from grizli.pipeline import auto_script
        auto_script.field_rgb(root, HOME_PATH=None, ds9=ds9, scale_ab=23)
        ds9.set('rgb lock colorbar')
        ds9.set('frame lock wcs')
                    
    if False:
        ds9 = None
        kwargs = {'ds9':ds9, 'mag_limit':[24,27], 'galfit_flux_limit':20, 'any_limit':18, 'point_limit':17, 'bkg_kwargs':{'order_npix':64}} 
    else:
        kwargs['ds9'] = ds9
    
    # Galfit not working on AWS
    if os.path.exists('/home/ec2-user'):
        print('Running on AWS: turn off Galfit')
        kwargs['galfit_flux_limit'] = None
        kwargs['refine_brightest'] = False
        
    if 'channels' in kwargs:
        channels = kwargs['channels']
    
    models = {}
                 
    for ch in channels:
        try:
            self = golfir.model.ImageModeler(root=root, lores_filter=ch, **kwargs) 
        except:
            continue
            
        patch_file = f'{self.root}_patch.fits'
        if os.path.exists(patch_file):
            tab = grizli.utils.read_catalog(patch_file)
        else:
            tab = self.generate_patches(**kwargs)
    
        for k in ['patch_arcmin', 'rd_patch', 'patch_id']:
            if k in kwargs:
                _ = kwargs.pop(k)
            
        N = len(tab)
        if (not use_patches):
            N = 1
            patches = [None]
        else:
            patches = [(tab['ra'][i], tab['dec'][i]) for i in range(N)]
            
        for i in range(N):
            try:
                rd_patch = patches[i]
                if ch > 'ch2':
                    if 'galfit_flux_limit' in kwargs:
                        if kwargs['galfit_flux_limit'] is not None:
                            kwargs['galfit_flux_limit'] = 100
                    else:
                        kwargs['galfit_flux_limit'] = None
                        
                self.run_full_patch(rd_patch=rd_patch, patch_arcmin=tab['patch_arcmin'][i], patch_id=tab['patch_id'][i], **kwargs)# ds9=None, patch_id=0, mag_limit=24, galfit_flux_limit=None, match_geometry=False, **kwargs)
                models[ch] = self
            except:
                pass
    
    # Add to HTML
    if sync_results:
        resid_images = glob.glob(f"{root}_*_*png")
        if len(resid_images) > 0:
            resid_images.sort()
        
            fp = open(f'{root}.irac.html','a')
            fp.write(f'\n\n<h2> Model ({time.ctime()})</h2>')
        
            extra = glob.glob(f'{root}_irac_phot.fits')
            extra += glob.glob(f'{root}*model.fits')
            extra += glob.glob(f'{root}*patch*')
            extra += glob.glob(f'{root}*components.fits')
            extra.sort()
            if len(extra) > 0:
                extra.sort()
                fp.write('\n<pre>')
                for file in extra:
                    fp.write(f'\n<a href={file}>{file}</a>')
                fp.write('\n</pre>')
            
            for resid in resid_images:
                fp.write(f'\n<br><tt>{resid}</tt>')
                fp.write(f'\n<br><a href="{resid}"><img src="{resid}" height=200px /> </a>')
        
            fp.close()
        
        # Sync
        os.system(f'aws s3 sync ./ s3://grizli-v1/Pipeline/{root}/IRAC/ '
                  f' --exclude "*" --include "{root}*model.fits"'
                  f' --include "{root}*irac*phot.fits"'
                  f' --include "{root}*components.fits"'
                  f' --include "{root}_*_*png"'
                  f' --include "{root}*patch*"'
                  f' --include "{root}.irac.html"'
                  ' --acl public-read')
              
    #########################
    # Manual position
    if False:
        for ch in ['ch1', 'ch2', 'ch3', 'ch4']:
            
            if False:
                scl = {'ch1':1.0, 'ch2':1.02, 'ch3':1.0, 'ch4':1.0}
                os.system('rm *model.fits *components.fits')
                psf_file = glob.glob(f'../AvgPSF/irac_{ch}_*[md]_psf.fits')[-1]
                print(psf_file)
                psf = pyfits.open(psf_file)
                psf[0].data = psf[0].data**(scl[ch])
                psf.writeto(f'{root}-{ch}-0.1.psfr_avg.fits', overwrite=True)

            try:
                self = golfir.model.ImageModeler(root=root, lores_filter=ch, **kwargs) 
            except:
                pass

            self.run_full_patch(rd_patch=None, patch_id=0, **kwargs)
            
            if 0:
                from photutils import (HanningWindow, TukeyWindow, CosineBellWindow, SplitCosineBellWindow, TopHatWindow)
                
    return models
    