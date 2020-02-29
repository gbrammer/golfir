import os
import glob
import numpy as np
from grizli import utils
from imp import reload
import matplotlib.pyplot as plt
import astropy.io.fits as pyfits
import astropy.wcs as pywcs
import astropy.units as u
import drizzlepac

from . import irac

# try:
#     import grizli.ds9
#     ds9 = grizli.ds9.DS9()
# except:
#     ds9 = None
    
def fetch_irac(root='j003528m2016', path='./'):
    ipac = utils.read_catalog(path+'{0}_ipac.fits'.format(root))
    
    # Overlaps
    ext = np.array([e.split('/')[0] for e in ipac['externalname']])
    ext_list = np.unique(ext[ipac['with_hst']])
    
    inst = np.array([e.split(' ')[0] for e in ipac['wavelength']])
    
    keep = (inst == 'IRAC') & (utils.column_values_in_list(ext, ext_list))
    
    if keep.sum() == 0:
        return False
                
    so = np.argsort(ipac['externalname'][keep])
    idx = np.arange(len(ipac))[keep][so]
    
    print('\n\n==================\n Fetch {0} files \n==================\n\n'.format(keep.sum()))
    
    N = keep.sum()
    
    for ix, i in enumerate(idx):
        if ('pbcd/' in ipac['externalname'][i]) | ('_maic.fits' in ipac['externalname'][i]):
            continue
            
        cbcd = glob.glob(ipac['externalname'][i].replace('_cbcd.fits', '_xbcd.fits.gz'))
        if len(cbcd) > 0:
            print('CBCD ({0:>4} / {1:>4}): {2}'.format(ix, N, cbcd[0]))
            continue
            
        xbcd = glob.glob(ipac['externalname'][i].replace('_bcd.fits', '_xbcd.fits.gz'))
        if len(xbcd) > 0:
            print('XBCD ({0:>4} / {1:>4}): {2}'.format(ix, N, xbcd[0]))
            continue
            
        out = '{0}_{1:05d}.zip'.format(root, i)
        print('Fetch ({0:>4} / {1:>4}): {2}'.format(ix, N, out))
        if not os.path.exists(out):
            os.system('wget -O {1} "{2}"'.format(root, out, ipac['accessWithAnc1Url'][i]))
            
            os.system('unzip -n {0}'.format(out))
            print('')
    
    if len(glob.glob(f'{root}*.zip')) == 0:
        return False
                
    # for i in idx:
    #     out = '{0}_{1:05d}.zip'.format(root, i)
    #     os.system('unzip -n {0}'.format(out))
    
    return ipac[keep]
    
def get_wcslist(query='r*', skip=10):
    import glob
    import astropy.io.fits as pyfits
    import astropy.wcs as pywcs
    files = glob.glob('{0}/ch*/bcd/SPITZER_I*cbcd.fits'.format(query))
    files += glob.glob('{0}/ch*/bcd/SPITZER_M*bcd.fits'.format(query))
        
    files.sort()
    
    if skip < 0:
        skip = int(np.ceil(len(files)/-skip))
        
    wcslist = []
    for i, file in enumerate(files[::skip]):
        print(i+1, len(files)//skip)
        wcslist.append(pywcs.WCS(pyfits.open(file)[0].header, relax=True))
    
    return wcslist
    
def run_irac(root='j003528m2016'):
        
    HOME = os.getcwd()
    
    if not os.path.exists(root):
        try:
            os.mkdir(root)
        except:
            pass

    os.chdir(root)
    fetch_irac(root)
        
def process_all(root):
            
    ipac = utils.read_catalog('../{0}_ipac.fits'.format(root))
    ch1 = ipac['wavelength'] == 'IRAC 3.6um'
    if False & ((root == 'j095532p1809') | (ch1.sum() > 150)):
        pixfrac, pix, kernel = 0.2, 0.5, 'point'
    else:
        pixfrac, pix, kernel = 0.2, 0.5, 'square'

    pixfrac, pix, kernel = 0.2, 0.5, 'square'
    
    if True:
        wcslist = get_wcslist(skip=10)
        out_hdu = utils.make_maximal_wcs(wcslist, pixel_scale=pix, theta=0, pad=5, get_hdu=True, verbose=True)
    else:
        out_hdu = None
    
    # Table copy
    phot = utils.read_catalog('{0}_phot.fits'.format(root))
    for ch in [1,2]:
        phot['irac_ch{0}_flux'.format(ch)] = -99.
        phot['irac_ch{0}_err'.format(ch)] = -99.
    
    phot.write('{0}_irac_phot.fits'.format(root), overwrite=True)
    
    # Make sure pixels align
    ref_file = glob.glob('{0}-f1*_drz_sci.fits'.format(root))[-1]
    ref_hdu = pyfits.open(ref_file)[0].header
    ref_filter = utils.get_hst_filter(ref_hdu).lower()
    
    ref_wcs = pywcs.WCS(ref_hdu)
    ref_rd = ref_wcs.all_pix2world(np.array([[-0.5, -0.5]]), 0).flatten()
    target_phase = np.array([0.5, 0.5])#/(pix/0.1)
    for k in ['RADESYS', 'LATPOLE', 'LONPOLE']:
        out_hdu.header[k] = ref_hdu[k]
        
    # Shift CRVAL to same tangent point
    out_wcs = pywcs.WCS(out_hdu.header)
    out_xy = out_wcs.all_world2pix(np.array([ref_wcs.wcs.crval]), 1).flatten()
    out_hdu.header['CRVAL1'], out_hdu.header['CRVAL2'] = tuple(ref_wcs.wcs.crval)
    out_hdu.header['CRPIX1'], out_hdu.header['CRPIX2'] = tuple(out_xy)
    
    # Align integer pixel phase
    out_wcs = pywcs.WCS(out_hdu.header)
    out_xy = out_wcs.all_world2pix(np.array([ref_rd]), 0).flatten()
    xy_phase = out_xy - np.floor(out_xy)
    new_crpix = out_wcs.wcs.crpix - (xy_phase - target_phase)
    out_hdu.header['CRPIX1'], out_hdu.header['CRPIX2'] = tuple(new_crpix)
    out_wcs = pywcs.WCS(out_hdu.header)
    
    if root == 'sxds':
        pixfrac, pix, kernel = 0.333, 0.5, 'square'
        out_hdu = None
        aor_ids = glob.glob('r3788????')
        aor_ids.sort()
        #aor_ids = aor_ids[:3]
        ch = 1
        aors_ch = {}
        psf_size = 32
        p = 0.5
        plt.ioff()
    else:
        aor_ids = glob.glob('r????????')
        aor_ids = None
    
    if root == 'j122656p2355':
        min_frametime=10
    else:
        min_frametime=20
                
    channels = [1,2]
    aors_ch = {}
    plt.ioff()
    
    for ch in channels:
        aors_ch[ch] = irac.process_all(channel='ch{0}'.format(ch), output_root=root, driz_scale=pix, kernel=kernel, pixfrac=pixfrac, wcslist=None, pad=0, out_hdu=out_hdu, aor_ids=aor_ids, flat_background=False, two_pass=True, min_frametime=min_frametime)
        
        psf_size=20
        for p in [0.1, pix]:
            irac.mosaic_psf(output_root=root, target_pix=p, channel=ch, aors=aors_ch[ch], kernel=kernel, pixfrac=pixfrac, size=psf_size, native_orientation=False)
            plt.close('all')
            
        psf_size=30
        p = 0.1
        irac.mosaic_psf(output_root=root, target_pix=p, channel=ch, aors=aors_ch[ch], kernel=kernel, pixfrac=pixfrac, size=psf_size, native_orientation=True)
        plt.close('all')
        
        del(aors_ch[ch])
        
    plt.ion()
    
    if False:
        res = self.drizzle_output(driz_scale=1.0, pixfrac=0.5, kernel='square', theta=0, wcslist=None, out_hdu=None, pedestal=self.pedestal, pad=10, psf_coords=None)
        res2 = self.drizzle_output(driz_scale=1.0, pixfrac=0.5, kernel='square', theta=0, wcslist=None, out_hdu=None, pedestal=self.med2d.T, pad=10, psf_coords=None)
        
    # Make PSFs
    psf_size=20
    for ch in channels:
        for p in [0.1, pix]:
            irac.mosaic_psf(output_root=root, target_pix=p, channel=ch, aors=aors_ch[ch], kernel=kernel, pixfrac=pixfrac, size=psf_size, native_orientation=True)
            plt.close('all')
            
            
    if False:
        files = glob.glob('*ch{0}-{1:.1f}*psfr.fits'.format(ch, p))
        files.sort()
        avg = None
        for file in files: 
            im = pyfits.open(file)
            if avg is None:
                wht = im[0].data != 0
                avg = im[0].data*wht
            else:
                wht_i = im[0].data != 0
                avg += im[0].data*wht_i
                wht += wht_i
        
        avg = avg/wht
        avg[wht == 0] = 0
        pyfits.writeto('{0}-ch{1}-{2:0.1f}.psfr_avg.fits'.format(root, ch, p), data=avg, header=im[0].header, overwrite=True)
        
    os.chdir(HOME)
    return aors_ch1, aors_ch2
    
    #######################
    # Redrizzle
    aors = aors_ch[1]
    
    all_wcs = []
    for aor in aors_ch1:
        all_wcs.extend(aors_ch1[aor].wcs)

    for aor in aors_ch2:
        all_wcs.extend(aors_ch2[aor].wcs)
    
    channel = aors[list(aors.keys())[0]].channel
            
    wcslist=None
    #wcslist = [pywcs.WCS(pyfits.open('j015020m1006-ir_drz_sci.fits')[0].header)]
    out_hdu = utils.make_maximal_wcs(all_wcs, pixel_scale=pix, theta=0, pad=0, get_hdu=True, verbose=True)
        
    for i, aor in enumerate(aors):
        print(i, aor, aors[aor].N)
        aors[aor].drz = aors[aor].drizzle_simple(wcslist=wcslist, driz_scale=pix, theta=0, kernel=kernel, out_hdu=out_hdu, pad=0, pixfrac=pixfrac, force_pedestal=False)#'point')
    
    output_root = root
    for i, aor in enumerate(aors):
        sci, wht, ctx, head, w = aors[aor].drz
        
        root_i = '{0}-{1}-{2}'.format(output_root, aor, channel)
        pyfits.writeto('{0}_drz_sci.fits'.format(root_i), data=sci, header=head, overwrite=True)
        pyfits.writeto('{0}_drz_wht.fits'.format(root_i), data=wht, header=head, overwrite=True)
        
        if i == 0:
            h0 = head
            h0['NDRIZIM'] = aors[aor].N - aors[aor].nskip
            num = sci*wht
            den = wht
        else:
            h0['EXPTIME'] += head['EXPTIME']
            h0['NDRIZIM'] += aors[aor].N - aors[aor].nskip
            num += sci*wht
            den += wht
    
    drz_sci = num/den
    drz_sci[den == 0] = 0
    drz_wht = den
    
    irac_root = '{0}-{1}'.format(output_root, channel)
    
    pyfits.writeto('{0}_drz_sci.fits'.format(irac_root), data=drz_sci, header=h0, overwrite=True)
    pyfits.writeto('{0}_drz_wht.fits'.format(irac_root), data=drz_wht, header=h0, overwrite=True)
    
    # Incorporate into catalog
    from grizli.pipeline import auto_script, photoz
    from grizli import prep
    import astropy.units as u
    from grizli import utils
    import numpy as np
    import astropy.io.fits as pyfits
    
    args = auto_script.get_yml_parameters(local_file='{0}.auto_script.yml'.format(root), copy_defaults=False, verbose=True, skip_unknown_parameters=True)
    multiband_catalog_args = args['multiband_catalog_args'].copy()
    
    tab = utils.read_catalog('{0}_phot.fits'.format(root))
    seg_data = pyfits.open('{0}-ir_seg.fits'.format(root))[0].data
    seg_data = np.cast[np.int32](seg_data)

    aseg, aseg_id = seg_data, tab['number']
    source_xy = tab['ra'], tab['dec'] #, aseg, aseg_id
    
    stars = (tab['mag_auto'] > 18) & (tab['mag_auto'] < 24) & (tab['flux_radius'] > 0.8) & (tab['flux_radius'] < 1.6)
    irac_tab = {}
    for ch in [1,2]:
        root_ch = '{0}-ch{1}'.format(root, ch)

        multiband_catalog_args['bkg_params'] = {'bh': 64, 'bw': 64, 'fh': 3, 'fw': 3, 'pixel_scale': 0.1}

        multiband_catalog_args['phot_apertures'] = [2*u.arcsec, 2*2.44*u.arcsec, 10*u.arcsec]
        apcorr = [1.21, 1.23][ch-1]*2
        
        irac_tab[ch] = prep.make_SEP_catalog(root=root_ch,
                  threshold=1.8, 
                  rescale_weight=True,
                  err_scale=-np.inf,
                  get_background=True,
                  save_to_fits=False, source_xy=source_xy,
                  phot_apertures=multiband_catalog_args['phot_apertures'],
                  bkg_mask=multiband_catalog_args['bkg_mask'],
                  bkg_params=multiband_catalog_args['bkg_params'],
                  use_bkg_err=False)
        
        
        irac_tab[ch].apcorr = apcorr # (irac_tab[ch]['FLUX_APER_7']/irac_tab[ch]['FLUX_APER_5'])[stars]
    
    #
    for ch in [1,2]:
        tab['irac_ch{0}_flux'.format(ch)] = irac_tab[ch]['FLUX_APER_0']*irac_tab[ch].apcorr
        tab['irac_ch{0}_err'.format(ch)] = irac_tab[ch]['FLUXERR_APER_0']*irac_tab[ch].apcorr
        
        bad = (~np.isfinite(tab['irac_ch{0}_flux'.format(ch)])) | (~np.isfinite(tab['irac_ch{0}_err'.format(ch)])) | (tab['irac_ch{0}_err'.format(ch)] < 0)
        tab['irac_ch{0}_flux'.format(ch)][bad] = -99
        tab['irac_ch{0}_err'.format(ch)][bad] = -99
        
    tab.write('{0}_irac_phot.fits'.format(root), overwrite=True)
    
    #
    from grizli.pipeline import photoz
    import numpy as np

    total_flux = 'flux_auto_fix'
    total_flux = 'flux_auto' # new segmentation masked SEP catalogs
    object_only = False
    
    self, cat, zout = photoz.eazy_photoz(root+'_irac', object_only=object_only, apply_prior=False, beta_prior=True, aper_ix=1, force=True, get_external_photometry=False, compute_residuals=False, total_flux=total_flux)
    
    #####################################################
    import scipy.ndimage as nd
    from stsci.convolve import convolve2d
    from skimage.transform import warp
    from skimage.transform import SimilarityTransform
    from skimage.morphology import dilation
    from skimage.morphology import binary_dilation

    from photutils import create_matching_kernel
    from photutils import (HanningWindow, TukeyWindow, CosineBellWindow,
                           SplitCosineBellWindow, TopHatWindow)
    
    #window = CosineBellWindow(alpha=0.8)
    window = HanningWindow()
    
    # Images
    ref_file = glob.glob('{0}-f1*_drz_sci.fits'.format(root)) 
    ref_file += glob.glob('{0}-ir*sci.fits'.format(root))
    
    ref_file.sort()
    ref_file = ref_file[-1]
    
    hst_im = pyfits.open(ref_file)
    ref_filter = utils.get_hst_filter(hst_im[0].header).lower()
    hst_wht = pyfits.open(ref_file.replace('_sci', '_wht'))
    hst_psf = pyfits.open('{0}-{1}_psf.fits'.format(root, 'f160w'))[1].data
    
    if os.path.exists('star_psf.fits'):
        st = pyfits.open('star_psf.fits')
        hst_psf = st['OPSF','F160W'].data*1
        
    hst_psf /= hst_psf.sum()
    
    hst_seg = pyfits.open(root+'-ir_seg.fits')[0]
    hst_ujy = hst_im[0].data*hst_im[0].header['PHOTFNU'] * 1.e6
    hst_wcs = pywcs.WCS(hst_im[0].header)

    phot = utils.read_catalog('{0}irac_phot.fits'.format(root))
    
    # Watershed segmentation dilation
    from skimage.morphology import watershed
    from astropy.convolution.kernels import Gaussian2DKernel
    kern = Gaussian2DKernel(5).array
    
    med = np.median(hst_im[0].data[(hst_wht[0].data > 0) & (hst_seg.data == 0)])
    hst_conv = convolve2d(hst_im[0].data-med, kern, fft=1)
    hst_var = 1/hst_wht[0].data
    wht_med = np.percentile(hst_wht[0].data[hst_wht[0].data > 0], 5)
    hst_var[hst_wht[0].data < wht_med] = 1/wht_med
    hst_cvar = convolve2d(hst_var, kern**2, fft=1)
    
    
    xi = np.cast[int](np.round(phot['xpeak']))
    yi = np.cast[int](np.round(phot['ypeak']))
    markers = np.zeros(hst_im[0].data.shape, dtype=int)
    markers[yi, xi] = phot['number']
    #waterseg = watershed(-hst_conv, markers, mask=((hst_wht[0].data > 0) & (hst_conv/np.sqrt(hst_cvar) > 1)))
    waterseg = watershed(-hst_conv, markers, mask=(hst_conv/np.sqrt(hst_cvar) > 1))
    waterseg[waterseg == 0] = hst_seg.data[waterseg == 0]
    
    ############ Parameters
    avg_psf=None
    avg_psf = pyfits.open('{0}-ch{1}-0.1.psfr_avg.fits'.format(root, ch))[0].data
    avg_kern = None #np.ones((5,5))
    hst_psf_offset = [2,2]
    hst_psf_size = hst_psf.shape[0]//2
    
    # Bright stars 
    # bright_limits = [16,19]
    try:
        _ = bright_limits
        bright_ids = None
    except:
        bright_limits = [16,19]
        bright_ids = None
        
    bright_sn = 3
        
    ###############
    
    if ch == 'mips1':
        irac_im = pyfits.open('{0}-{1}_drz_sci.fits'.format(root, ch))[0]
        irac_wht = pyfits.open('{0}-{1}_drz_wht.fits'.format(root, ch))[0].data
        irac_psf_obj = irac.MipsPSF()
        pf = 10
        
        column_root = 'mips_24'
        phot['{0}_flux'.format(column_root)] = -99.
        phot['{0}_err'.format(column_root)] = -99.
        phot['{0}_bright'.format(column_root)] = 0
        
        bright_limits = None
        #ERR_SCALE = 0.1828 # From residuals
        ch_label = ch
        ERR_SCALE = 1.
        
    else:
        irac_im = pyfits.open('{0}-ch{1}_drz_sci.fits'.format(root, ch))[0]
        irac_wht = pyfits.open('{0}-ch{1}_drz_wht.fits'.format(root, ch))[0].data
        irac_psf_obj = irac.IracPSF(ch=ch, scale=0.1, verbose=True, avg_psf=avg_psf)
        try:
            ir_tab = utils.read_catalog('{0}-ch{1}.cat.fits'.format(root, ch))
            ERR_SCALE = ir_tab.meta['ERR_SCALE']
        except:
            ERR_SCALE = 1.
                    
        try:
            irac_wcs = pywcs.WCS(irac_im.header)
            pscale = utils.get_wcs_pscale(irac_wcs)
            pf = int(np.round(pscale/0.1))
        except:
            pf = 5

        column_root = 'irac_ch{0}'.format(ch)        
        phot['{0}_flux'.format(column_root)] = -99.
        phot['{0}_err'.format(column_root)] = -99.
        phot['{0}_bright'.format(column_root)] = 0
        ch_label = 'ch{0}'.format(ch)
        
    # HST psf in same dimensions as IRAC
    tf=None
    rd = hst_wcs.wcs.crval
    _psf, _, _ = irac_psf_obj.evaluate_psf(ra=rd[0], dec=rd[1], 
                                  min_count=1, clip_negative=True, 
                                  transform=tf)
                                  
    hst_psf_full = np.zeros_like(_psf)
    sh_irac = hst_psf_full.shape
    hslx = slice(sh_irac[0]//2+hst_psf_offset[0]-hst_psf_size, sh_irac[0]//2+hst_psf_offset[0]+hst_psf_size)
    hsly = slice(sh_irac[0]//2+hst_psf_offset[1]-hst_psf_size, sh_irac[0]//2+hst_psf_offset[1]+hst_psf_size)
    hst_psf_full[hsly, hslx] += hst_psf
        
    irac_wcs = pywcs.WCS(irac_im.header)

    # Shifts
    if False:
        idx, dr = phot.match_to_catalog_sky(ir_tab)
        mat = (dr.value < 1) & (ir_tab['mag_auto'] > 17) & (ir_tab['flux_radius'] > 2) & (ir_tab['flux_radius'] < 7)
        hst_irac_x, hst_irac_y = irac_wcs.all_world2pix(phot['ra'][idx], phot['dec'][idx], 0)
        xoff = ir_tab['x'] - hst_irac_x
        yoff = ir_tab['y'] - hst_irac_y
        
        from astropy.modeling.models import Polynomial2D
        from astropy.modeling.fitting import LinearLSQFitter
        fitter = LinearLSQFitter() 
        m2 = Polynomial2D(degree=3)
        m0 = irac_wcs.wcs.crval
        mx = fitter(m2, ir_tab['ra'][mat]-m0[0], ir_tab['dec'][mat]-m0[1], -xoff[mat])
        my = fitter(m2, ir_tab['ra'][mat]-m0[0], ir_tab['dec'][mat]-m0[1], -yoff[mat])
        
        if False:
            rx_i = mx(ir_tab['ra']-m0[0], ir_tab['dec']-m0[1])
            ry_i = my(ir_tab['ra']-m0[0], ir_tab['dec']-m0[1])
        
    else:
        mx = my = None
        
    # weight in IRAC frame
    hst_wht_i = hst_wht[0].data[::pf, ::pf]
    ll = np.cast[int](irac_wcs.all_world2pix(hst_wcs.all_pix2world(np.array([[-0.5, -0.5]]), 0), 0)).flatten()
    irac_mask = np.zeros(irac_im.data.shape)
    isly = slice(ll[1], ll[1]+hst_wht_i.shape[0])
    islx = slice(ll[0], ll[0]+hst_wht_i.shape[1])
    irac_wht[isly, islx] *= hst_wht_i > 0
    irac_sivar = np.sqrt(irac_wht)/ERR_SCALE
    
    tf = None

    ############################################################
    ############################################################
    ############################################################
                         
    if get_pan:
        
        rd_pan = np.cast['float'](ds9.get('pan fk5').split())
        
        xy_irac = np.cast[int](np.round(irac_wcs.all_world2pix(np.array([rd_pan]), 0))).flatten()
        
        ll_irac = xy_irac-Npan
        ll_hst_raw = hst_wcs.all_world2pix(irac_wcs.all_pix2world(np.array([xy_irac-Npan])-0.5, 0), 0)
        ll_hst = np.cast[int](np.ceil(ll_hst_raw)).flatten()
        
        #Npan = 256
        slx = slice(ll_hst[0], ll_hst[0]+2*Npan*pf)
        sly = slice(ll_hst[1], ll_hst[1]+2*Npan*pf)
    
        if False:
            print('Dilate segmentation image')
            
            seg_sl = hst_seg.data[sly, slx]*1
            ids = np.unique(seg_sl)[1:]
            in_seg = utils.column_values_in_list(phot['number'], ids) & (phot['mag_auto'] < 26)
            in_seg &= np.isfinite(phot[ref_filter+'_fluxerr_aper_1'])
            in_seg &= phot[ref_filter+'_fluxerr_aper_1'] > 0
            
            so = np.argsort(phot['mag_auto'][in_seg])
            
            sizes = np.clip(np.cast[int](np.sqrt(phot['npix'])), 2, 64)
            so = np.argsort(sizes[in_seg])[::-1]
            
            for iter in range(3):
                for id, mag, si in zip(phot['number'][in_seg][so], phot['mag_auto'][in_seg][so], sizes[in_seg][so]):
                    #break
                    #si = np.clip(np.cast[int](1.8**(26-np.clip(mag, 18, 25))), 3, 32)
                    print('{0} {1:.2f} {2}'.format(id, mag, si))
                    seg_dilate = np.zeros((si, si))
                    seg_dilate[si//2,:] = 1
                    seg_dilate[:,si//2] = 1

                    dil_id = binary_dilation(seg_sl == id, seg_dilate)
                    seg_sl[(seg_sl == 0) & dil_id] = id
            
            ds9.frame(10)
            ds9.view(seg_sl, header=utils.to_header(hst_wcs.slice((sly, slx))))
        else:
            #seg_sl = hst_seg.data[sly, slx]*1
            seg_sl = waterseg[sly, slx]*1
            ds9.frame(10)
            ds9.view(seg_sl, header=utils.to_header(hst_wcs.slice((sly, slx))))
                        
    ids = np.unique(seg_sl)[1:]
    N = len(ids)
    #rx, ry = 0, 0

    islx = slice(ll_irac[0], ll_irac[0]+2*Npan)
    isly = slice(ll_irac[1], ll_irac[1]+2*Npan)
    
    print('Compute object models (N={0})'.format(N))
    
    source_psf = True
    if source_psf:
        # Evaluate PSF at each position
        _A = []
        hst_slice = hst_ujy[sly, slx]
        for i, id in enumerate(ids):
            print(i, id)
            ix = phot['number'] == id
            
            if mx is not None:
                rx_i = mx(phot['ra'][ix]-m0[0], phot['dec'][ix]-m0[1])
                ry_i = my(phot['ra'][ix]-m0[0], phot['dec'][ix]-m0[1])
                tf = np.array([rx_i, ry_i]) #{'translation':(rx_i, ry_i)}

            else:
                #tf = {'translation':(rx, ry)}
                tf = None
                
            _ = irac_psf_obj.evaluate_psf(ra=phot['ra'][ix], dec=phot['dec'][ix], min_count=1, clip_negative=True, transform=tf)

            irac_psf, psf_exptime, psf_count = _                              
            if window is -1:
                psf_kernel = irac_psf
            else:
                psf_kernel = create_matching_kernel(hst_psf_full, irac_psf, window=window)

            # Extra factor for rebinning pixel grids, assuming 5x oversampling
            if avg_kern is not None:
                psf_kernel = convolve2d(psf_kernel, avg_kern, mode='constant', fft=1, cval=0.)
            else:
                psf_kernel *= pf**2
                
            _Ai = convolve2d(hst_slice*(seg_sl == id), psf_kernel,
                              mode='constant', fft=1)[::pf, ::pf].flatten()
            _A.append(_Ai)                 
        
        _A = np.array(_A)
            
    else:
        try:
            _ = rd_pan
        except:
            rd_pan = hst_wcs.wcs.crval

        #if tf is None:
        #    tf = {'translation':(rx, ry)}
            
        _ = irac_psf_obj.evaluate_psf(ra=rd_pan[0], dec=rd_pan[1], 
                                      min_count=1, clip_negative=True, 
                                      transform=tf)

        irac_psf, psf_exptime, psf_count = _                              

        if window is -1:
            psf_kernel = irac_psf
        else:
            psf_kernel = create_matching_kernel(hst_psf_full, irac_psf, window=window)
        
        # Extra factor for rebinning pixel grids, assuming 5x oversampling
        if avg_kern is not None:
            psf_kernel = convolve2d(psf_kernel, avg_kern, mode='constant', cval=1., fft=1)
        else:
            psf_kernel *= pf**2
            
        _A = []
        for i, id in enumerate(ids):
            print(i, id)
            _Ai = convolve2d(hst_slice*(seg_sl == id), psf_kernel,
                              mode='constant', fft=1)[::pf, ::pf].flatten()
            _A.append(_Ai)
            
        _A = np.array(_A) #np.array([convolve2d(hst_ujy[sly, slx]*(seg_sl == id), psf_kernel, mode='constant', fft=1)[::np, ::np].flatten() for id in ids])
    
    y = irac_im.data[isly, islx].flatten()
    sivar = (irac_sivar[isly, islx]).flatten()
    
    # Convolved with HST fluxes
    h_sm = _A.sum(axis=0).reshape(seg_sl[::pf,::pf].shape)
        
    # Normalize so that coefficients are directly uJy
    Anorm = _A.sum(axis=1)
    keep = Anorm > 0
    _A = (_A[keep,:].T/Anorm[keep]).T
    ids = ids[keep]
    Anorm = Anorm[keep]
    N = keep.sum()
    
    # Background components
    from numpy.polynomial.chebyshev import chebgrid2d as polygrid2d
    from numpy.polynomial.hermite import hermgrid2d as polygrid2d

    #poly_order = 3
    #poly_pix_size = 64
    try:
        poly_order = np.clip(int(np.round(2*Npan/poly_pix_size)), 2, 11)
    except:
        poly_order = np.clip(int(np.round(2*Npan/64)), 2, 11)
        
    x = np.linspace(-1, 1, 2*Npan)
    _Abg = []
    c = np.zeros((poly_order, poly_order))
    for i in range(poly_order):
        for j in range(poly_order):
            c*=0
            c[i][j] = 1.e-3
            _Abg.append(polygrid2d(x, x, c).flatten())
            
    _Abg = np.array(_Abg)
    
    # Redo transform fit from here
    _Af = np.vstack([_Abg, _A])  
            
    if bright_limits is not None:
        bright = (phot['mag_auto'][ids-1] < bright_limits[0]) #& (phot['flux_radius'][ids-1] < 1.5)  
        bright |= (phot['mag_auto'][ids-1] < bright_limits[1]) & (phot['flux_radius'][ids-1] < 3.5)  
        
        if bright_ids is not None:
            for ii in bright_ids:
                bright[ids == ii] = True
                
        h_bright = _A[bright,:].T.dot(Anorm[bright])
    else:
        h_bright = 0.
        bright = None

    # Border
    border = np.ones((2*Npan*pf, 2*Npan*pf))
    yp, xp = np.indices((border.shape))
    R = np.sqrt((xp+pf-Npan*pf)**2+(yp+pf-Npan*pf)**2)
    border[1:-1,1:-1] = 0
    #border = (R > Npan*np)*1
    b_conv = convolve2d(border, psf_kernel,
                      mode='constant', fft=1, cval=1)[::pf, ::pf].flatten()
    b_conv = b_conv/b_conv.max()
                      
    msk = sivar > 0
    msk &= h_bright*sivar < bright_sn
    msk &= b_conv < 0.1
    msk2d = msk.reshape(h_sm.shape)
    _Ax = (_Af[:,msk]*(sivar)[msk]).T
    _yx = (y*sivar)[msk]
    
    #bad = ~np.isfinite(_Ax)
    print('Least squares')
    
    _x = np.linalg.lstsq(_Ax, _yx, rcond=-1)

    h_model = _Af.T.dot(_x[0]).reshape(h_sm.shape)
    
    Nbg = _Abg.shape[0]
    h_bg = _Af[:Nbg].T.dot(_x[0][:Nbg]).reshape(h_sm.shape)
    
    # IRAC image
    ds9.frame(11)
    ds9.view(h_sm, header=utils.get_wcs_slice_header(irac_wcs, islx, isly))

    ds9.frame(12)
    ds9.view(irac_im.data, header=irac_im.header)
    
    ds9.frame(13)
    ds9.view(msk2d*h_model, header=utils.get_wcs_slice_header(irac_wcs, islx, isly))

    ds9.frame(14)
    ds9.view(msk2d*(irac_im.data[isly, islx]-h_model), header=utils.get_wcs_slice_header(irac_wcs, islx, isly))
    ds9.frame(15)
    ds9.view(msk2d*(irac_im.data[isly, islx]-h_model), header=utils.get_wcs_slice_header(irac_wcs, islx, isly))
    get_pan = False

    ################### Alignment
    t0 = np.array([0,0,0,1])*np.array([1,1,1,100.])

    #t0 = np.array([0,0,0])*np.array([1,1,1])
    
    from scipy.optimize import minimize 
    args = (irac_im.data[isly, islx], h_model, sivar.reshape(h_model.shape)*msk2d, 0)
    
    _res = minimize(_obj_shift, t0, args=args, method='powell')
    
    args = (irac_im.data[isly, islx], h_model, sivar.reshape(h_model.shape), 1)
    tfx, warped = _obj_shift(_res.x, *args)
    tf = tfx*1

    ds9.frame(16)
    ds9.view(msk2d*(irac_im.data[isly, islx]-warped), header=utils.get_wcs_slice_header(irac_wcs, islx, isly))   
    
    # Shifts
    if False:
        yp, xp = np.indices(irac_im.data.shape)
        yp = yp[isly, islx]
        xp = xp[isly, islx]
        _, xw = _obj_shift(_res.x,  irac_im.data[isly, islx], xp*1., sivar.reshape(h_model.shape), 1)
        _, yw = _obj_shift(_res.x,  irac_im.data[isly, islx], yp*1., sivar.reshape(h_model.shape), 1)
        
    for i in range(_A.shape[0]):
        print(i)
        _A[i,:] = irac.warp_image(tf, _A[i,:].reshape(h_model.shape)).flatten()
    
    ####################### Redo transformed model
    _Af = np.vstack([_Abg, _A])  
        
    # Bright stars
    if bright is not None:
        h_bright = _A[bright,:].T.dot(Anorm[bright])
    else:
        h_bright = 0
        
    msk = sivar > 0
    msk &= b_conv < 0.1
    msk &= h_bright*sivar < bright_sn
    msk2d = msk.reshape(h_sm.shape)

    _Ax = (_Af[:,msk]*sivar[msk]).T
    _yx = (y*sivar)[msk]
    
    #bad = ~np.isfinite(_Ax)
    print('Least squares')
    
    _x = np.linalg.lstsq(_Ax, _yx, rcond=-1)

    h_model = _Af.T.dot(_x[0]).reshape(h_sm.shape)
    
    Nbg = _Abg.shape[0]
    h_bg = _Af[:Nbg].T.dot(_x[0][:Nbg]).reshape(h_sm.shape)
    
    # IRAC image
    ds9.frame(12)
    ds9.view(irac_im.data, header=irac_im.header)
    
    ds9.frame(13)
    ds9.view(msk2d*h_model, header=utils.get_wcs_slice_header(irac_wcs, islx, isly))

    ds9.frame(14)
    ds9.view(msk2d*(irac_im.data[isly, islx]-h_model), header=utils.get_wcs_slice_header(irac_wcs, islx, isly))
    get_pan = False
    
    ############################################################
    ############################################################
    ############################################################
    
    if False:
        # compute error scale
        resid = (irac_im.data[isly, islx]-h_model).flatten()*sivar
        rmask = msk & (h_model.flatten()*sivar < 1)
        ERR_SCALE_i = utils.nmad(resid[rmask]) 
        print('ERR_SCALE_i: {0:.3f}'.format(ERR_SCALE_i))
        ERR_SCALE *= ERR_SCALE_i
        
        irac_sivar /= ERR_SCALE_i
        sivar = (irac_sivar[isly, islx]).flatten()
        
        msk = sivar > 0
        msk &= b_conv < 0.1
        msk &= h_bright*sivar < bright_sn
        msk2d = msk.reshape(h_sm.shape)

        _Ax = (_Af[:,msk]*sivar[msk]).T
        _yx = (y*sivar)[msk]

        #bad = ~np.isfinite(_Ax)
        print('Least squares')
        _x = np.linalg.lstsq(_Ax, _yx, rcond=-1)
        h_model = _Af.T.dot(_x[0]).reshape(h_sm.shape)
        
        ds9.frame(14)
        ds9.view(msk2d*(irac_im.data[isly, islx]-h_model), header=utils.get_wcs_slice_header(irac_wcs, islx, isly))
        
        if False:
            ii = np.where(ids == id)[0][0]
            m0 = (_A[ii,:]*_x[0][Nbg+ii]).reshape(h_model.shape)
            ds9.view(msk2d*(irac_im.data[isly, islx] - (h_model - m0)), header=head)
            ds9.view(_A[ii,:].reshape(h_model.shape)*10, header=head)
        
        if False:
            # Aperture photometry on the residuals
            import sep
            wsl = irac_wcs.slice((isly, islx))  
            x, y = wsl.all_world2pix(phot['ra'][ids-1], phot['dec'][ids-1], 0)
            im_err = 1./sivar.reshape(h_model.shape)
            im_err[~np.isfinite(im_err)] = 100
            
            ap_radius = 1.5 # arcsec
            ap_flux, ap_err, flg = sep.sum_circle((irac_im.data[isly, islx]-h_model*0), x, y, ap_radius/(pf/10.), err=im_err, var=None, mask=(~msk2d), maskthresh=0.0, segmap=None, seg_id=None,  bkgann=None, gain=None, subpix=5)

            resid_flux, resid_err, flg = sep.sum_circle((irac_im.data[isly, islx]-h_model), x, y, ap_radius/(pf/10.), err=im_err, var=None, mask=(~msk2d), maskthresh=0.0, segmap=None, seg_id=None,  bkgann=None, gain=None, subpix=5)
            
            # Apcorr from PSF
            yp, xp = np.indices(irac_psf.shape)
            x0 = (xp*irac_psf).sum()
            y0 = (yp*irac_psf).sum()
            Rp = np.sqrt((xp-x0)**2+(yp-y0)**2).flatten()*0.1
            so = np.argsort(Rp)
            cog = np.cumsum(irac_psf.flatten()[so])
            apcorr = 1./np.interp(ap_radius, Rp[so], cog)
            
            ##
            wcs = pywcs.WCS(sci[0].header)
            clean = sci[0].data - model[0].data
            msk = (model[0].data*np.sqrt(wht[0].data) < 1) & (wht[0].data > 0)
            msk *= model[0].data > 0
            
            err_scale = utils.nmad((clean*np.sqrt(wht[0].data))[msk])
            im_err = 1/np.sqrt(wht[0].data)*err_scale
            im_err[wht[0].data == 0] = 100
            
            x, y = wcs.all_world2pix(phot['ra'], phot['dec'], 0)
            ap_radius = 1.5 # arcsec
            ap_flux, ap_err, flg = sep.sum_circle(clean, x, y, ap_radius/(pf/10.), err=im_err, var=None, mask=None, maskthresh=0.0, segmap=None, seg_id=None,  bkgann=None, gain=None, subpix=5)
            
            
    ####################### Save results 
    full_model = irac_im.data*0
    full_model[isly, islx][msk2d] = h_model[msk2d]
    pyfits.writeto('{0}-{1}_model.fits'.format(root, ch_label), data=full_model, header=irac_im.header, overwrite=True)
    
    ds9.frame(15)
    ds9.view(full_model, header=irac_im.header)
    
    if root == 'j094952p1707':
        id = 111
        m0 = (_A[ii,:]*_x[0][Nbg+ii]).reshape(h_model.shape)
        alma_model = irac_im.data*0
        alma_model[isly, islx] = m0
        pyfits.writeto('{0}-{1}_alma.fits'.format(root, ch_label), data=alma_model, header=irac_im.header, overwrite=True)
        
    flux = _x[0][Nbg:Nbg+N]*1

    covar = np.matrix(np.dot(_Ax.T, _Ax)).I.A
    err = np.sqrt(covar.diagonal())[Nbg:Nbg+N]
    
    # Clip negative
    bad = flux < -2*err
    flux[bad] = -99
    err[bad] = -99
    
    # Ignore background in covariance .... doesn't make much difference
    # covar2 = np.matrix(np.dot(_Ax[:,Nbg:].T, _Ax[:,Nbg:])).I.A
    # err2 = np.sqrt(covar2.diagonal())
    
    phot.meta['{0}_ERR_SCALE'.format(column_root.upper())] = ERR_SCALE
    
    if 'mips' not in column_root:
        phot['{0}_flux'.format(column_root)][ids-1] = flux
        phot['{0}_err'.format(column_root)][ids-1] = err
        nexp, expt = irac_psf_obj.get_exposure_time(phot['ra'], phot['dec'], verbose=True)
        phot['{0}_nexp'.format(column_root)] = nexp
        phot['{0}_exptime'.format(column_root)] = expt
    else:
        
        # Fix bug accounting for IRAC/MIPS pixel scales
        # if 'ORIGPIX' not in irac_im.header:
        #     print('Scale MIPS fluxes')
        #     flux *= (2.50/1.223)**2
        #     err *= (2.50/1.223)**2
            
        phot['{0}_flux'.format(column_root)][ids-1] = flux*irac_psf_obj.apcorr
        phot['{0}_err'.format(column_root)][ids-1] = err*irac_psf_obj.apcorr

        phot['F325'] = phot['mips_24_flux']
        phot['E325'] = phot['mips_24_err']
                        
    if bright is not None:
        try:
            phot['{0}_bright'.format(column_root)][ids[bright]-1] = 1
        except:
            phot['{0}_bright'.format(column_root)] = 0
            phot['{0}_bright'.format(column_root)][ids[bright]-1] = 1

    # Neighbors
    if False:
        import astropy.units as u
        pos = np.array([phot['ra'], phot['dec']]).T
        pos = (pos - np.median(pos,  axis=0))
        tree = scipy.spatial.cKDTree(pos)
        dn, nn = tree.query(pos, k=2)
        phot['dr_nn'] = dn[:,1]*3600*u.arcsec
        phot['dr_nn'].format = '.2f'
        
        phot['dmag_nn'] = phot['mag_auto'] - phot['mag_auto'][nn[:,1]]
        phot['dr_nn'].format = '.2f'

        phot['id_nn'] = phot['number'][nn[:,1]]
        
        plt.scatter(23.9-2.5*np.log10(flux), err*np.sqrt(expt[ids-1]/3600.), alpha=0.2, c=phot['dr_nn'][ids-1], vmin=0, vmax=2) 
        
    phot.write('{0}irac_phot.fits'.format(root), overwrite=True)
    
        
def _obj_shift(transform, data, model, sivar, ret):
    from skimage.transform import SimilarityTransform
    
    if len(transform) == 2:
        tf = np.array(transform)/10.
    elif len(transform) == 3:
        tf = np.array(transform)/np.array([10,10,1])
    elif len(transform) == 4:
        tf = np.array(transform)/np.array([10,10, 1., 100])
        
    if len(model) == 2:
        warped = irac.warp_image(tf, model[0])#[::np,::np]
        warped = convolve2d(warped, avg_kern, mode='constant', fft=1, cval=0.)[::np,::np]
        _Am = (np.vstack([warped.flatten()[None, :], model[1]]))
        _Ax = (_Am*sivar.flatten()).T
        _yx = (data*sivar).flatten()

        _x = np.linalg.lstsq(_Ax, _yx, rcond=-1)
        #_a = _x[0][0]
        #warped *= _a
        warped = _Am.T.dot(_x[0]).reshape(data.shape)
        
        msk = ((warped > msk_scale*data) | (warped > 0.2*warped.max())).flatten()
        #msk = ((warped > msk_scale*data)).flatten()
        print(msk.sum())
        _x = np.linalg.lstsq(_Ax[msk,:], _yx[msk], rcond=-1)
        #_a = _x[0][0]
        #warped *= _a
        warped = _Am.T.dot(_x[0]).reshape(data.shape)
        
    else:
        warped = irac.warp_image(tf, model)
        
    if ret == 1:
        return tf, warped
        
    chi2 = (data-warped)**2*sivar**2
    mask = sivar > 0
    chi2s = chi2[mask].sum()
    print(tf, chi2s)
    return chi2s
    
def fit_point_source(t0, poly_order=5):
    Npan = 20

    # if False:
    #     avg_psf = pyfits.open('j041632m2407-ch1-0.1.psfr_avg.fits')[0].data
    #     avg_psf /= avg_psf.sum()
    # 
    #     irac_psf_obj = irac.IracPSF(ch=ch, scale=0.1, verbose=True, avg_psf=avg_psf)
    
    Npan = int(irac_psf_obj.psf_arrays['shape'][0]/10)
        
    rd_pan = np.cast['float'](ds9.get('pan fk5').split())
    
    xy_irac = np.cast[int](np.round(irac_wcs.all_world2pix(np.array([rd_pan]), 0))).flatten()
    
    ll_irac = xy_irac-Npan
    ll_hst = np.cast[int](np.floor(hst_wcs.all_world2pix(irac_wcs.all_pix2world(np.array([xy_irac-Npan]), 0), 0))).flatten()
    
    #Npan = 256
    slx = slice(ll_hst[0], ll_hst[0]+2*Npan*5)
    sly = slice(ll_hst[1], ll_hst[1]+2*Npan*5)
    
    islx = slice(ll_irac[0], ll_irac[0]+2*Npan)
    isly = slice(ll_irac[1], ll_irac[1]+2*Npan)
    
    _ = irac_psf_obj.evaluate_psf(ra=rd_pan[0], dec=rd_pan[1], 
                                  min_count=1, clip_negative=False, 
                                  transform=None)

    irac_psf, psf_exptime, psf_count = _                              
    #irac_psf = irac_psf_obj.psf_arrays['image'][ic,:].reshape((300,300))
    
    wht = pyfits.open('{0}-ch{1}_drz_wht.fits'.format(root, ch))[0].data

    sivar = np.sqrt(wht[isly, islx])
    
    #poly_order=5
    x = np.linspace(-1, 1, 2*Npan)
    _Abg = []
    c = np.zeros((poly_order, poly_order))
    for i in range(poly_order):
        for j in range(poly_order):
            c*=0
            c[i][j] = 1.e-3
            _Abg.append(polygrid2d(x, x, c).flatten())
    _Abg = np.array(_Abg)
    
    args = (irac_im.data[isly, islx], (irac_psf, _Abg), sivar, 0)
    
    #t0 = np.array([0,0,0,1])*np.array([1,1,1,100.])
    
    _res = minimize(_obj_shift, t0, args=args, method='powell')
    
    args = (irac_im.data[isly, islx], (irac_psf, _Abg), sivar, 1)
    _shift, _psf = _obj_shift(_res.x, *args)

    ds9.frame(19)
    ds9.view(_psf, header=utils.get_wcs_slice_header(irac_wcs, islx, isly))
    
    ds9.frame(18)
    ds9.view(irac_im.data[isly, islx], header=utils.get_wcs_slice_header(irac_wcs, islx, isly))

    psf_model = irac_im.data*0
    psf_model[isly, islx] += _psf

    ds9.frame(20)
    aa = 1
    ds9.view(irac_im.data-psf_model*aa, header=irac_im.header)
    ds9.set('pan to {0} {1} fk5'.format(rd_pan[0], rd_pan[1]))
    
    return _psf
    
def alma_source():
    # ALMA source
    if False:
        # zero-out nearby object
        ix = np.where(ids == 1019)[0][0]
        _xx = _x[0]
        _xx[Nbg+ix] = 0
        h_model2 = _Af.T.dot(_xx).reshape(h_sm.shape)
        
        full_model = irac_im.data*0
        full_model[isly, islx] += h_model2
        pyfits.writeto('{0}-ch{1}_model.fits'.format(root, ch), data=full_model, header=irac_im.header, overwrite=True)
        
        #########
        import sep
        rg, dg = 205.5360413, 9.478936196
        rq, dq = 205.5337798, 9.477328082
        rr, dd = rg, dg
        
        phot = utils.read_catalog('{0}irac_phot_apcorr.fits'.format(root))

        irac_im = pyfits.open('{0}-ch{1}_drz_sci.fits'.format(root, ch))[0]
        irac_wht = pyfits.open('{0}-ch{1}_drz_wht.fits'.format(root, ch))[0].data

        irac_wcs = pywcs.WCS(irac_im.header)
        full_model = pyfits.open('{0}-ch1_model.fits'.format(root))[0].data
        
        irac_psf, psf_nexp = irac.evaluate_irac_psf(ch=ch, scale=0.1, 
                         ra=rg, dec=dg)
                                 
        xy_ap = irac_wcs.all_world2pix(np.array([[rg, dg]]), 0).T
        apers = np.append(4, np.arange(2,10.1, 0.5))
        _res = sep.sum_circle(irac_im.data-full_model, xy_ap[0], xy_ap[1], apers, var=1/irac_wht)
        
        sh_psf = irac_psf.shape
        _res_psf = sep.sum_circle(irac_psf, [sh_psf[1]//2-1], [sh_psf[0]//2-1], apers*5, var=irac_psf*0.+1)
        
        ch1_flux = (_res[0]/_res_psf[0])[0]
        ch1_err = (_res[1]/_res_psf[0])[0]
        
        parent = ['ch1']
        wave = [3.6]
        fnu = [1.7927988]
        fnu_sum = [1.7927988]
        efnu = [0.2448046]
        pixscale = [0.5]
        
        parent += ['ch2']
        wave += [4.5]
        fnu += [3.060]
        fnu_sum += [3.06]
        efnu += [0.3203]
        pixscale += [0.5]
        width = [0.25]*2
        
        alm = utils.read_catalog('alma.info', format='ascii')
        alm['wave'] = 2.99e8/alm['CRVAL3']/1.e-6*u.micron
        alm['wave'].format = '.1f'
        
        # ALMA images
        files = glob.glob('ALMA/*spw2?.mfs*fits')
        files += glob.glob('ALMA/*spw??_*fits')
        files.sort()
        for file in files:
            print(file)
            alma = pyfits.open(file)
            alma_wcs = pywcs.WCS(alma[0].header)
            pscale = np.abs(alma_wcs.wcs.cdelt[0]*3600.)
            alma_xy = alma_wcs.all_world2pix(np.array([[rr, dd, 0, 0]]), 0).flatten()
            yp, xp = np.indices(np.squeeze(alma[0].data).shape)
            R = np.sqrt((xp-alma_xy[0])**2 + (yp-alma_xy[1])**2) < 1.3/pscale
            R &= np.isfinite(np.squeeze(alma[0].data))
            fnu.append(np.squeeze(alma[0].data)[R].max()*1.e6)
            fnu_sum.append(np.squeeze(alma[0].data)[R].sum()*1.e6)
            wave.append(2.999e8/alma[0].header['CRVAL3']*1.e6)
            width.append(alma[0].header['CDELT3']/alma[0].header['CRVAL3']*wave[-1])
            parent.append(file)
            msk = np.isfinite(alma[0].data)
            efnu.append(utils.nmad(alma[0].data[msk])*1.e6)
            pixscale.append(pscale)
        
        plt.errorbar(wave, fnu, xerr=width, yerr=efnu, marker='o', alpha=0.5, linestyle='None', color='k')
        #plt.scatter(wave, fnu, c=np.log10(np.array(width)/np.array(wave)), marker='o', alpha=0.8, zorder=100, cmap='plasma_r')
        
        #plt.errorbar(wave, fnu_sum, efnu, marker='o', alpha=0.1, linestyle='None')
        plt.xlabel(r'$\lambda_\mathrm{obs}$, $\mu\mathrm{m}$')
        plt.ylabel(r'flux, $\mu$Jy')
        plt.loglog()
        
        width_lim = [0.1]*3
        wave_lim = [1.25, 1.05, 0.81]
        
        efnu_lim = [np.median(phot['f{0}w_etot_1'.format(b)][phot['f{0}w_etot_1'.format(b)] > 0]) for b in ['125','105','814']]
        np.random.seed(3)
        fnu_lim = list(np.random.normal(size=3)*np.array(efnu_lim))
        plt.errorbar(wlim, fnu_lim, efnu_lim, marker='v', alpha=0.5, linestyle='None', color='k')
        parent_lim = ['f125w_limit','f105w_limit','f814w_limit']
        
        tab = utils.GTable()
        tab['wave'] = wave + wave_lim; tab['wave'].unit = u.micron
        tab['dw'] = width + width_lim; tab['dw'].unit = u.micron
        tab['fnu'] = fnu + fnu_lim; tab['fnu'].unit = u.microJansky
        tab['efnu'] = efnu + efnu_lim; tab['efnu'].unit = u.microJansky
        tab['parent'] = parent + parent_lim
        tab.meta['RA'] = (rg, 'Target RA')
        tab.meta['DEC'] = (dg, 'Target Dec')
        tab.meta['TIME'] = (time.ctime(), 'Timestamp of file generation')
        tab['wave'].format = '.1f'
        tab['dw'].format = '.1f'
        for c in tab.colnames:
            if 'fnu' in c:
                tab[c].format = '.3f'
                
        jname = utils.radec_to_targname(ra=rg, dec=dg, round_arcsec=(0.01, 0.01*15), precision=2, targstr='j{rah}{ram}{ras}.{rass}{sign}{ded}{dem}{des}.{dess}', header=None)
        so = np.argsort(tab['wave'])
        
        tab[so].write('alma_serendip_{0}.fits'.format(jname), overwrite=True)
        tab[so].write('alma_serendip_{0}.ipac'.format(jname), format='ascii.ipac', overwrite=True)
        
        
        if False:
            plt.scatter(1.25, 3*np.median(phot['f125w_etot_1'][phot['f125w_etot_1'] > 0]), marker='v', color='k', alpha=0.8)
            plt.scatter(1.05, 3*np.median(phot['f105w_etot_1'][phot['f105w_etot_1'] > 0]), marker='v', color='k', alpha=0.8)
            plt.scatter(0.81, 3*np.median(phot['f814w_etot_1'][phot['f814w_etot_1'] > 0]), marker='v', color='k', alpha=0.8)
        
        plt.xlim(0.4, 5000)
        plt.ylim(01.e-5, 1.e4)
        plt.gca().set_yticks(10**np.arange(-5, 4.1))
        plt.grid()
        
        ### FSPS
        import fsps
        import astropy.constants as const
        import astropy.units as u
        from astropy.cosmology import Planck15
        
        try:
            print(sp.params['dust2'])
        except:
            sp = fsps.StellarPopulation(zcontinuous=True, dust_type=2, sfh=4, tau=0.1)
        dust2 = 6
        z = 1.5
        tage = 0.2
        
        sp.params['dust2'] = dust2
        sp.params['add_neb_emission'] = True
        wsp, flux = sp.get_spectrum(tage=tage, peraa=False)
        
        flux = (flux/(1+z)*const.L_sun).to(u.erg/u.s).value*1.e23*1.e6
        dL = Planck15.luminosity_distance(z).to(u.cm).value
        flux /= 4*np.pi*dL**2
        
        yi = np.interp(3.6e4, wsp*(1+z), flux)
        
        label = 'z={0:.1f} tage={1:.1f}'.format(z, tage) + '\n'+ 'dust2={0:.1f}, logM={1:.1f} SFR={2:.1f}'.format(dust2, np.log10(sp.stellar_mass*fnu[0]/yi), sp.sfr_avg()*fnu[0]/yi)
        plt.plot(wsp*(1+z)/1.e4, flux/yi*fnu[0], alpha=0.6, zorder=-1, label=label)
        plt.legend()
        plt.tight_layout()
        plt.savefig('alma_serendip_{0}.pdf'.format(jname))
        
        # Cubes
        alma = pyfits.open('ALMA/member.uid___A001_X1296_X96a.Pisco_sci.spw27.cube.I.pbcor.fits')
        
        cube_files = glob.glob('ALMA/*cube*fits')
        cube_files.sort()
        alma = pyfits.open(cube_files[ci])
        alma_wcs = pywcs.WCS(alma[0].header)
        pscale = np.abs(alma_wcs.wcs.cdelt[0]*3600.)
        alma_xy = alma_wcs.all_world2pix(np.array([[rr, dd, 0, 0]]), 0).flatten()
        yp, xp = np.indices(alma[0].data[0,0,:,:].shape)
        R = np.sqrt((xp-alma_xy[0])**2 + (yp-alma_xy[1])**2) < 0.5/pscale
        fin = np.isfinite(alma[0].data)
        alma[0].data[~fin] = 0
        clip = alma[0].data[0,:,:]*R
        fnu_max = clip.max(axis=1).max(axis=1)
        
        freq = (np.arange(alma_wcs._naxis[2])+1-alma_wcs.wcs.crpix[2])*alma_wcs.wcs.cdelt[2] + alma_wcs.wcs.crval[2]
        dv = np.gradient(freq)/freq*3.e5
        ng = 20./np.median(dv)
        sm = nd.gaussian_filter(fnu_max, ng)
        plt.plot(freq/1.e9, sm)
        plt.plot(freq/1.e9, fnu_max, color='k', alpha=0.1)
        
    if False:
        # candidate
        ix = ids == 588
        coeffs = _x[0]*(~ix)
        h_m2 = _A.T.dot(coeffs).reshape(h_sm.shape)
        ds9.frame(16)
        ds9.view(irac_im.data[isly, islx]-h_m2, header=utils.get_wcs_slice_header(irac_wcs, islx, isly))

def fsps_observed_frame(sp, tage=0.2, z=5, cosmology=None, stellar_mass=1.e11):
    from astropy.cosmology import Planck15
    import astropy.constants as const
    import astropy.units as u
    import numpy as np
    
    if cosmology == None:
        cosmology = Planck15
        
    wave, flux = sp.get_spectrum(tage=tage, peraa=False)
    
    flux = (flux*(1+z)*const.L_sun).to(u.erg/u.s).value*1.e23*1.e6
    dL = Planck15.luminosity_distance(z).to(u.cm).value
    flux /= 4*np.pi*dL**2
    flux *= stellar_mass / sp.stellar_mass
    
    return wave*(1+z)*u.Angstrom, flux*u.microJansky
    
    import fsps
    import numpy as np
    import matplotlib.pyplot as plt
    from eazy.igm import Inoue14
    igm = Inoue14()
    mpl_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    sp = fsps.StellarPopulation(zcontinuous=True, sfh=4, tau=0.1, logzsol=-0.5)
    
    sp.params['sfh'] = 4
    sp.params['tau'] = 0.1
    sp.params['add_neb_emission'] = True  
    sp.params['dust_type'] = 2
    sp.params['dust2'] = 0.1
    
    z = 5
    ic=0

    mass = 1.e9
    ages = [0.1, 0.2, 0.3]
    
    colors = sns.light_palette(mpl_colors[ic], n_colors=len(ages)+1, reverse=True)
    
    for ia, tage in enumerate(ages):
        w, f = fsps_observed_frame(sp, tage=tage, z=z, stellar_mass=mass)
        
        igmz = igm.full_IGM(z, w.value)
        plt.plot(w/1.e4, 23.9-2.5*np.log10(f.value*igmz), label=r'z={0:.1f} tage={1:.1f} $\tau_\mathrm{{{{SFH}}}}$={2:.0f} xxx M={3:.1f}, $\tau_\mathrm{{{{V}}}}$={4:.1f}'.format(z, tage, sp.params['tau']*1000, np.log10(mass), sp.params['dust2']).replace('xxx ', '\n'), alpha=0.5, color=colors[ia])
    
    plt.semilogx()
    plt.legend()
    plt.ylim(30, 22)
    plt.xlim(0.1, 18)

def run_galfit(id=7739):
    
    #print(i, id)
    tf = None
    psf_offset = 1.e-6
    wpower = 0.05
    use_psf=False
    psf_offset, wpower = 0, -1
    from grizli.galfit import galfit
    
    ix = phot['number'] == id
          
    _ = irac_psf_obj.evaluate_psf(ra=phot['ra'][ix], dec=phot['dec'][ix], min_count=10, clip_negative=False, transform=tf)

    irac_psf, psf_exptime, psf_count = _                              
    irac_psf += psf_offset
    irac_psf *= (irac_psf > 0)
    if wpower > 0:
        coswindow = CosineBellWindow(alpha=1)
        irac_psf *= coswindow(irac_psf.shape)**wpower
    
    irac_psf /= irac_psf.sum()
    
    segmap = ((seg_sl == id) | (seg_sl == 0))[pf//2::pf,pf//2::pf]
    #segmap = ((seg_sl == id))[2::5,2::5]
    segmap *= msk2d
    
    _c = _x[0]*1.
    _c[Nbg + np.arange(len(ids))[ids == id]] = 0
    
    _Af = np.vstack([_Abg, _A])  
    h_model = _Af.T.dot(_c).reshape(h_sm.shape)
    ydata = irac_im.data[isly, islx] - h_model
    ivar = (irac_sivar[isly, islx])**2
        
    if use_psf:
        components = [galfit.GalfitPSF(), galfit.GalfitSky()]
    else:
        components = [galfit.GalfitSersic(), galfit.GalfitSky()]
        #components = [galfit.GalfitSersic(disk=True), galfit.GalfitSersic(), galfit.GalfitSky()]
        #components = [galfit.GalfitExpdisk(), galfit.GalfitSky()]
    
    #components = components[:1]
    
    # Don't include sky in model
    if len(components) > 1:
        components[-1].output = 1

    wsl = irac_wcs.slice((isly, islx))  
    xy = wsl.all_world2pix(np.array([phot['ra'][ix], phot['dec'][ix]]).T, 1).flatten()
    for i in range(len(components)-1):
        components[i].pdict['pos'] = list(xy)

    components[0].pdict['mag'] = phot['mag_auto'][ix][0]+1.1
    
    if False:
        fit_ids = [7126,7127,7129,7130,7641,7580,7579,7580,7578,7327,6530,6632,7821]
        components = [galfit.GalfitSky()]
        segmap = (seg_sl == 0)[2::5,2::5]

        _c = _x[0]*1.
    
        _Af = np.vstack([_Abg, _A])  

        for id_i in fit_ids:
            ix = phot['number'] == id_i
            _c[Nbg + np.arange(len(ids))[ids == id_i]] = 0
            segmap |= ((seg_sl == id_i))[2::5,2::5]
            components += [galfit.GalfitSersic()]
            xy = wsl.all_world2pix(np.array([phot['ra'][ix], phot['dec'][ix]]).T, 1).flatten()
            components[-1].pdict['pos'] = list(xy)
            components[-1].pdict['mag'] = phot['mag_auto'][ix][0]+1.1
    
        h_model = _Af.T.dot(_c).reshape(h_sm.shape)
        ydata = irac_im.data[isly, islx] - h_model
    
    #components[0].pfree['mag'] = 0
    
    #components[0].pdict['R_e'] = 0.1

    ivar = 1/(1/ivar+(0.02*ydata)**2)

    if use_psf:
        gf = galfit.Galfitter.fit_arrays(ydata, ivar, segmap*1, irac_psf, psf_sample=5, id=1, components=components, recenter=False, exptime=0)
    else:
        gf = galfit.Galfitter.fit_arrays(ydata, ivar, segmap*1, irac_psf[pf//2::pf,pf//2::pf], psf_sample=1, id=1, components=components, recenter=False, exptime=0)
    
    chi2 = float(gf['ascii'][3].split()[3][:-1])
    
    if chi2 < 50:

        _A[ids == id,:] = gf['model'].data.flatten()/gf['model'].data.sum()
        
        if recompute_coeffs:
            _Af = np.vstack([_Abg, _A])  

            # Bright stars
            if bright is not None:
                h_bright = _A[bright,:].T.dot(Anorm[bright])
            else:
                h_bright = 0

            msk = sivar > 0
            msk &= h_bright*sivar < bright_sn
            msk2d = msk.reshape(h_sm.shape)

            _Ax = (_Af[:,msk]*sivar[msk]).T
            _yx = (y*sivar)[msk]

            #bad = ~np.isfinite(_Ax)
            print('Least squares')

            _x = np.linalg.lstsq(_Ax, _yx, rcond=-1)
            h_model = _Af.T.dot(_x[0]).reshape(h_sm.shape)
        else:
            h_model += gf['model'].data
            
        #Nbg = _Abg.shape[0]
        h_bg = _Af[:Nbg].T.dot(_x[0][:Nbg]).reshape(h_sm.shape)

        ds9.frame(13)
        ds9.view(h_model*msk2d, header=utils.get_wcs_slice_header(irac_wcs, islx, isly))

        ds9.frame(14)
        ds9.view(msk2d*(irac_im.data[isly, islx]-h_model), header=utils.get_wcs_slice_header(irac_wcs, islx, isly))

def match_slice_to_shape(slparent, nparent):
    """
    Refine a slice for a parent array that might not contain a full child
    """
    
    nchild = slparent.stop - slparent.start
    
    if (slparent.start > nparent) | (slparent.stop < 1):
        null = slice(0,0)
        return null, null
        
    if slparent.start < 0:
        left = -slparent.start
    else:
        left = 0
    
    if slparent.stop > nparent:
        right = slparent.stop - nparent
    else:
        right = 0
    
    oslparent = slice(slparent.start+left, slparent.stop-right, slparent.step)
    oslchild= slice(left, nchild-right, slparent.step)
    return oslparent, oslchild
    
def effective_psf(log, rd=None, size=30, pixel_scale=0.1, pixfrac=0.2, kernel='square', recenter=False):
    """
    Drizzle effective PSF model given the oversampled model PRF
    """
    from photutils import (HanningWindow, TukeyWindow, CosineBellWindow, SplitCosineBellWindow, TopHatWindow)
    
    # r48106752/ch1/bcd/SPITZER_I1_48106752_0001_0000_2_cbcd.fits
    ch = int(log['file'][0].split("_")[1][-1])
    if __name__ == "__main__":
        import golfir
        _path = os.path.dirname(golfir.__file__)
    else:
        _path = os.path.dirname(__file__)
        
    h_file = os.path.join(_path, f'data/bcd_ch{ch}.header')
    sip_h = pyfits.Header.fromtextfile(h_file)
    
    N = len(log)
    
    #if rd is None:
    rd = np.mean(log['crval'][:N//2], axis=0)
    
    ipsf = pyfits.open(f'../AvgPSF/IRAC/apex_sh_IRACPC{ch}_col129_row129_x100.fits', relax=True)

    ipsf = pyfits.open(f'../AvgPSF/Cryo/apex_sh_IRAC{ch}_col129_row129_x100.fits', relax=True)

    #ipsf = pyfits.open(f'../AvgPSF/IRAC/IRACPC{ch}_col129_row129.fits')
    
    if 'PRFXRSMP' in ipsf[0].header:
        osamp = ipsf[0].header['PRFXRSMP']
    else:
        osamp = 100
        
    ish = ipsf[0].data.shape
    
    if recenter:
        # Centroid
        yp, xp = np.indices(ish)  
        pp = ipsf[0].data
        xc = int(np.round((pp*xp).sum()/pp.sum()))
        yc = int(np.round((pp*yp).sum()/pp.sum()))
        i0 = [xc, yc]
    else:
        i0 = [s//2 for s in ish]
        
    # Number of pixels to extract
    inp = np.min([xc, yc]) - osamp//2
    # Extra padding
    if osamp == 100:
        inp -= 20
    else:
        inp -= 1
        
    wcs_list = []
    sci_list = []
    wht_list = []
        
    wht_i = np.ones((256,256), dtype=np.float32)
    
    coords = [rd]
    cosd = np.cos(rd[1]/180*np.pi)
    for dx in np.linspace(-1, 1, 4):
        for dy in np.linspace(-1, 1, 4):
            if dx == dy == 0:
                continue
            
            delta = np.array([dx/cosd/60, dy/60])
            coords.append(rd+delta)
            
    for k in range(N):
        print('Parse file: {0}'.format(log['file'][k]))
        
        if 1:
            cd = log['cd'][k] 
            theta = np.arctan2(cd[1][0], cd[1][1])/np.pi*180
            print(k, theta)
            
        sip_h['CRPIX1'] = log['crpix'][k][0]
        sip_h['CRPIX2'] = log['crpix'][k][1]
        
        sip_h['CRVAL1'] = log['crval'][k][0]
        sip_h['CRVAL2'] = log['crval'][k][1]
        
        sip_h['LATPOLE'] = log['crval'][k][1]

        for i in range(2):
            for j in range(2):
                key = f'CD{i+1}_{j+1}'
                sip_h[key] = log['cd'][k][i,j]
        
        wcs_i = pywcs.WCS(sip_h, relax=True)
        wcs_i.pscale = utils.get_wcs_pscale(wcs_i)
        
        sci_i = np.zeros((256,256), dtype=np.float32)
        
        for coo in coords:
            try:
                xy = wcs_i.all_world2pix([coo], 0).flatten() + 0.5
                # if ch == 2:
                #     xy += 0.25
            except:
                print('wcs failed')
                continue
            
            xyp = np.cast[int](np.floor(xy))
            phase = np.cast[int](np.round((xy-xyp)*osamp - osamp/2.))
        
            oslx = slice(i0[0]-phase[0]-inp, i0[0]-phase[0]+inp+1, osamp)
            osly = slice(i0[1]-phase[1]-inp, i0[1]-phase[1]+inp+1, osamp)
            psf_sub = ipsf[0].data[osly, oslx]
            osh = psf_sub.shape
        
            nparent = 256
            slx0 = slice(xyp[0]-osh[1]//2, xyp[0]+osh[1]//2)
            sly0 = slice(xyp[1]-osh[0]//2, xyp[1]+osh[0]//2)  
        
            slpx, slcx = (match_slice_to_shape(slx0, nparent))    
            slpy, slcy = (match_slice_to_shape(sly0, nparent))    
            try:
                sci_i[slpy, slpx] += psf_sub[slcy, slcx]
            except:
                print('slice failed')
                
        wcs_list.append(wcs_i)
        sci_list.append(sci_i)
        wht_list.append((sci_i != 0).astype(np.float32))
        
        # ds9.frame(k)
        # ds9.view(sci_i, header=sip_h)
    
    if use_native_orientation:   
        cd = log['cd'][0] 
        theta = np.arctan2(cd[1][0], cd[1][1])/np.pi*180
    else:
        theta=0
    
    for k, coo in enumerate(coords):
        print('Drizzle coords: {0}'.format(coo))

        out_hdu = utils.make_wcsheader(ra=coo[0], dec=coo[1], size=size, pixscale=pixel_scale, theta=-theta, get_hdu=True)
        #out_h, out_wcs = _out
        out_wcs = pywcs.WCS(out_hdu.header)
        out_wcs.pscale = utils.get_wcs_pscale(out_wcs)

        if False:
            _drz = utils.drizzle_array_groups(sci_list, wht_list, wcs_list, 
                     outputwcs=out_wcs, pixfrac=pixfrac, kernel=kernel,
                     verbose=False)
            
            drz_psf = _drz[0] / _drz[0].sum()               
            pyfits.writeto(f'irsa_{pixel_scale}pix_ch{ch}_{k}_psf.fits', data=drz_psf, overwrite=True)
        else:   
            if k == 0:
                _drz = utils.drizzle_array_groups(sci_list, wht_list, wcs_list, 
                         outputwcs=out_wcs, pixfrac=pixfrac, kernel=kernel,
                         verbose=False)
            else:
                _ = utils.drizzle_array_groups(sci_list, wht_list, wcs_list, 
                         outputwcs=out_wcs, pixfrac=pixfrac, kernel=kernel,
                         verbose=False, data=_drz[:3])
                     
    sci, wht, ctx, head, w = _drz  
    coswindow = CosineBellWindow(alpha=1)(_drz[0].shape)**0.05

    drz_psf = (_drz[0]*coswindow) / (_drz[0]*coswindow).sum()     
              
    pyfits.writeto(f'irsa_{pixel_scale}pix_ch{ch}_psf.fits', data=drz_psf, overwrite=True)