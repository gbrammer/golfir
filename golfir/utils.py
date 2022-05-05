import os
import glob
import numpy as np

import matplotlib.pyplot as plt

import astropy.io.fits as pyfits
import astropy.wcs as pywcs
import astropy.units as u

try:
    import drizzlepac
except:
    print("(golfir.utils) Warning: failed to import drizzlepac")
    
try:
    import grizli.utils
except:
    print("(golfir.utils) Warning: failed to import grizli")

# try:
#     import grizli.ds9
#     ds9 = grizli.ds9.DS9()
# except:
#     ds9 = None

# `rcond` argument for np.linalg.lstsq  
LSTSQ_RCOND = None

# scipy.optimize.nnls kwargs
NNLS_KWARGS = {'maxiter': None}


def fetch_irac(root='j003528m2016', path='./', channels=['ch1','ch2','ch3','ch4'], force_hst_overlap=True):
    """
    Fetch IRAC and MIPS images listed in a `ipac.fits` catalog generated 
    by `mastquery`.
    """
    
    ipac = grizli.utils.read_catalog(path+'{0}_ipac.fits'.format(root))
    
    ch_trans = {'ch1': 'IRAC 3.6um',
                'ch2': 'IRAC 4.5um',
                'ch3': 'IRAC 5.8um',
                'ch4': 'IRAC 8.0um',
                'mips1': 'MIPS 24um',
                'mips2': 'MIPS 70um',
                'mips3': 'MIPS 160um'}
    
    # Overlaps
    ext = np.array([e.split('/')[0] for e in ipac['externalname']])
    if 'with_hst' in ipac.colnames:
        ext_list = np.unique(ext[ipac['with_hst']])
    else:
        ext_list = np.unique(ext)
        
    inst = np.array([e.split(' ')[0] for e in ipac['wavelength']])
    
    keep = ipac['ra'] < -1e10
    
    for ch in channels:
        if ch not in ch_trans:
            continue
        
        if ch.startswith('mips'):
            # Only EBCD MIPS files
            test = ipac['wavelength'] == ch_trans[ch]
            ebcd = grizli.utils.column_string_operation(ipac['externalname'], 
                                                  '_ebcd', 'count','or')
            keep |= test & ebcd
        else:
            keep |= ipac['wavelength'] == ch_trans[ch]
        
        #(inst == 'IRAC')
    
    # All exposures of an AOR that overlaps at at
    if force_hst_overlap:
        keep &= grizli.utils.column_values_in_list(ext, ext_list)
    
    # Explicit overlap for every exposure
    if force_hst_overlap > 1:
        if 'with_hst' in ipac.colnames:
            keep &= ipac['with_hst'] > 0

    if keep.sum() == 0:
        msg = """No matching exposures found for:
        root={0}
        channels={1}
        force_hst_overlap={2}"""
        print(msg.format(root, channels, force_hst_overlap))
        return False
                
    so = np.argsort(ipac['externalname'][keep])
    idx = np.arange(len(ipac))[keep][so]
    
    un, ct = np.unique(ipac['wavelength'][keep], return_counts=True)
    print('\n\n#==================\n# Fetch {0} files'.format(keep.sum()))
    print('#=================='.format(keep.sum()))
    for w, n in zip(un, ct):
        print('# {0:10} {1:>5}'.format(w, n))
    print('#==================\n\n'.format(keep.sum()))
    
    N = keep.sum()
    
    for ix, i in enumerate(idx):
        if ('pbcd/' in ipac['externalname'][i]) | ('_maic.fits' in ipac['externalname'][i]):
            continue
            
        cbcd = glob.glob(ipac['externalname'][i].replace('_cbcd.fits', '_xbcd.fits.gz'))
        if len(cbcd) > 0:
            print('CBCD ({0:>4} / {1:>4}): {2}'.format(ix, N, cbcd[0]))
            continue
            
        xbcd = glob.glob(ipac['externalname'][i].replace('_bcd.fits', '_xbcd.fits.gz'))
        xbcd += glob.glob(ipac['externalname'][i].replace('_ebcd.fits', '_xbcd.fits.gz'))
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


def _obj_shift(transform, data, model, sivar, ret):
    """
    Objective function for shifting the lores model
    """
    from skimage.transform import SimilarityTransform
    from . import irac
    
    if len(transform) == 2:
        tf = np.array(transform)/10.
    elif len(transform) == 3:
        tf = np.array(transform)/np.array([10,10,1])
    elif len(transform) == 4:
        tf = np.array(transform)/np.array([10,10, 1., 100])
        
    if len(model) == 2:
        ### Not working
        raise ValueError('model as list not implemented')
        
        # warped = irac.warp_image(tf, model[0])#[::np,::np]
        # #warped = convolve2d(warped, avg_kern, mode='constant', fft=1, cval=0.)[::np,::np]
        # warped = convolve_helper(warped, model[2], method='stsci', 
        #                          fill_scipy=False, cval=0.0)[::np,::np]
        # _Am = (np.vstack([warped.flatten()[None, :], model[1]]))
        # _Ax = (_Am*sivar.flatten()).T
        # _yx = (data*sivar).flatten()
        # 
        # _x = np.linalg.lstsq(_Ax, _yx, rcond=LSTSQ_RCOND)
        # #_a = _x[0][0]
        # #warped *= _a
        # warped = _Am.T.dot(_x[0]).reshape(data.shape)
        # 
        # msk = ((warped > msk_scale*data) | (warped > 0.2*warped.max())).flatten()
        # #msk = ((warped > msk_scale*data)).flatten()
        # print(msk.sum())
        # _x = np.linalg.lstsq(_Ax[msk,:], _yx[msk], rcond=LSTSQ_RCOND)
        # #_a = _x[0][0]
        # #warped *= _a
        # warped = _Am.T.dot(_x[0]).reshape(data.shape)
        
    else:
        warped = irac.warp_image(tf, model)
        
    if ret == 1:
        return tf, warped
        
    chi2 = (data-warped)**2*sivar**2
    mask = sivar > 0
    chi2s = chi2[mask].sum()
    print(tf, chi2s)
    return chi2s

RESCALE_KWARGS = {'order':1,'mode':'constant', 'cval':0., 
                  'preserve_range':False, 'anti_aliasing':True}

DRIZZLE_KWARGS = {'pixfrac':1.0, 'kernel':'square', 'verbose':False}

def resample_array(img, wht=None, pixratio=2, slice_if_int=True, int_tol=1.e-3, method='drizzle', drizzle_kwargs=DRIZZLE_KWARGS, rescale_kwargs=RESCALE_KWARGS, scale_by_area=False, verbose=False, blot_stepsize=-1, **kwargs):
    """
    Resample an image to a new grid.  If pixratio is an integer, just return a 
    slice of the input `img`.  Otherwise resample with `~drizzlepac` or `~resample`.
    """
    from grizli.utils import (make_wcsheader, drizzle_array_groups, 
                              blot_nearest_exact)

    from skimage.transform import rescale, resize, downscale_local_mean
    
    is_int = np.isclose(pixratio, np.round(pixratio), atol=int_tol)    
    if is_int & (pixratio > 1):
        # Integer scaling
        step = int(np.round(pixratio))
        if method.lower() == 'drizzle':
            _, win = make_wcsheader(ra=90, dec=0, size=img.shape, 
                                    pixscale=1., get_hdu=False, theta=0.)
            
            _, wout = make_wcsheader(ra=90, dec=0, size=img.shape, 
                                     pixscale=pixratio, get_hdu=False, 
                                     theta=0.)
                        
            if wht is None:
                wht = np.ones_like(img)
            
            _drz = drizzle_array_groups([img], [wht], [win], outputwcs=wout,
                                        **drizzle_kwargs)
            res = _drz[0]
            res_wht = _drz[1]
            method_used = 'drizzle'
            
        elif method.lower() == 'convolve':
            kern = np.ones((step,step))/step**2
            res = convolve_helper(img, kern)[step//2::step, step//2::step]
            res_wht = np.ones_like(res)
            method_used = 'convolve'
        elif slice_if_int:
            # Simple slice
            res = img[step//2::step, step//2::step]*1
            res_wht = np.ones_like(res)
            method_used = 'slice'
        else:
            # skimage downscale with averaging
            res = downscale_local_mean(img, (step, step), cval=0, clip=True)
            res_wht = np.ones_like(res)
            method_used = 'downscale'

    else:
        if method.lower() == 'drizzle':
            # Drizzle
            _, win = make_wcsheader(ra=90, dec=0, size=img.shape, 
                                    pixscale=1., get_hdu=False, theta=0.)
            
            _, wout = make_wcsheader(ra=90, dec=0, size=img.shape, 
                                     pixscale=pixratio, get_hdu=False, 
                                     theta=0.)
                        
            if wht is None:
                wht = np.ones_like(img)
            
            _drz = drizzle_array_groups([img], [wht], [win], outputwcs=wout,
                                        **drizzle_kwargs)
            res = _drz[0]
            res_wht = _drz[1]
            method_used = 'drizzle'
        
        elif method.lower() == 'blot':
            # Blot exact values
            _, win = make_wcsheader(ra=90, dec=0, size=img.shape, 
                                    pixscale=1., get_hdu=False, theta=0.)
            
            _, wout = make_wcsheader(ra=90, dec=0, size=img.shape, 
                                     pixscale=pixratio, get_hdu=False, 
                                     theta=0.)
            
            # Ones for behaviour around zeros
            res = blot_nearest_exact(img+1, win, wout, verbose=False, 
                                     stepsize=blot_stepsize, 
                                     scale_by_pixel_area=False, 
                                     wcs_mask=False, fill_value=0) - 1
                                     
            res_wht = np.ones_like(res)
            method_used = 'blot'
            
        elif method.lower() == 'rescale':
            res = rescale(img, 1./pixratio, **rescale_kwargs)
            res_wht = np.ones_like(res)
            method_used = 'rescale'
            
        else:
            raise ValueError("method must be 'drizzle', 'blot' or 'rescale'.")
    
    if scale_by_area:
        scale = 1./pixratio**2
    else:
        scale = 1
    
    if verbose:
        msg = 'resample_array x {4:.1f}: {0} > {1}, method={2}, scale={3:.2f}'
        print(msg.format(img.shape, res.shape, method_used, scale, pixratio))
            
    if not np.isclose(scale, 1, 1.e-4):
        res = res*scale
        res_wht = res_wht/scale**2
        
    #print(res_wht, res_wht.dtype, scale, res_wht.shape)
    #res_wht /= scale**2
    
    return res, res_wht


def fsps_observed_frame(sp, tage=0.2, z=5, cosmology=None, stellar_mass=1.e11):
    """
    Evaluate an SPS model spectrum in the observed frame, scaling by 
    stellar mass
    """
    from astropy.cosmology import Planck15
    import astropy.constants as const
    import astropy.units as u
    import numpy as np
    import seaborn as sns
    
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


def effective_psf(log, rd=None, size=30, pixel_scale=0.1, pixfrac=0.2, kernel='square', recenter=False, use_native_orientation=False, weight_exptime=True, subsample=4,  prf_file='IRAC/apex_sh_IRACPC{ch}_col129_row129_x100.fits'):
    """
    Drizzle effective PSF model given the oversampled model PRF
    
    ** not used, testing **
    """
    #from photutils import (HanningWindow, TukeyWindow, CosineBellWindow, SplitCosineBellWindow, TopHatWindow)
    try:    
        from photutils import (HanningWindow, TukeyWindow, 
                                CosineBellWindow, SplitCosineBellWindow, 
                                TopHatWindow)
    except:
        from photutils.psf.matching import (HanningWindow, TukeyWindow, 
                                CosineBellWindow, SplitCosineBellWindow, 
                                TopHatWindow)
    
    import grizli.utils
    
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
    
    if rd is None:
        rd = np.mean(log['crval'][:N//2], axis=0)
    
    #ipsf = pyfits.open(f'../AvgPSF/IRAC/apex_sh_IRACPC{ch}_col129_row129_x100.fits', relax=True)

    ipsf = pyfits.open(prf_file.format(ch=ch), relax=True)

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
        xc, yc = i0
        
    # Number of pixels to extract
    inp = np.min([xc, yc]) - osamp//2
    # Extra padding
    if osamp == 100:
        #inp -= 20
        inp = 1200
    else:
        inp -= 1
        
    wcs_list = []
    sci_list = []
    wht_list = []
        
    wht_i = np.ones((256,256), dtype=np.float32)
    
    coords = [rd]
    if subsample > 0:
        coords = []
        cosd = np.cos(rd[1]/180*np.pi)
        for dx in np.linspace(-1, 1, subsample):
            for dy in np.linspace(-1, 1, subsample):
                if dx == dy == 0:
                    continue
            
                delta = np.array([dx/cosd/60, dy/60])
                coords.append(rd+delta)
            
    for k in range(N):
        print('Parse file {0}: {1}'.format(k, log['file'][k]))
        
        if 0:
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
        wcs_i.pscale = grizli.utils.get_wcs_pscale(wcs_i)
        
        sci_i = np.zeros((256,256), dtype=np.float32)
        
        for coo in coords:
            try:
                xy = wcs_i.all_world2pix([coo], 0).flatten() #+ 0.5
                # if ch == 2:
                #     xy += 0.25
            except:
                print('wcs failed')
                continue
            
            #xyp = np.cast[int](np.floor(xy))
            #phase = np.cast[int](np.round((xy-xyp-0.5)*osamp))
            xyp = np.cast[int](np.round(xy))
            phase = np.cast[int](np.round((xy-xyp)*osamp))
        
            oslx = slice(i0[0]-phase[0]-inp, i0[0]-phase[0]+inp+1, osamp)
            osly = slice(i0[1]-phase[1]-inp, i0[1]-phase[1]+inp+1, osamp)
            psf_sub = ipsf[0].data[osly, oslx] #*coswindow
            osh = psf_sub.shape
        
            nparent = 256
            slx0 = slice(xyp[0]-osh[1]//2, xyp[0]+osh[1]//2+1)
            sly0 = slice(xyp[1]-osh[0]//2, xyp[1]+osh[0]//2+1)  
        
            slpx, slcx = (match_slice_to_shape(slx0, nparent))    
            slpy, slcy = (match_slice_to_shape(sly0, nparent))    
            try:
                sci_i[slpy, slpx] += psf_sub[slcy, slcx]
            except:
                print('slice failed', sci_i[slpy, slpx].shape, psf_sub[slcy, slcx].shape)
                
        wcs_list.append(wcs_i)
        sci_list.append(sci_i)
        wht = log['exptime'][k]**weight_exptime
        wht_list.append(((sci_i != 0)*wht).astype(np.float32))
    
    if use_native_orientation:   
        cd = log['cd'][0] 
        theta = np.arctan2(cd[1][0], cd[1][1])/np.pi*180
    else:
        theta=0
    
    for k, coo in enumerate(coords):
        print('Drizzle coords: {0}'.format(coo))

        out_hdu = grizli.utils.make_wcsheader(ra=coo[0], dec=coo[1], size=size, pixscale=pixel_scale, theta=-theta, get_hdu=True)
        #out_h, out_wcs = _out
        out_wcs = pywcs.WCS(out_hdu.header)
        out_wcs.pscale = grizli.utils.get_wcs_pscale(out_wcs)

        if False:
            _drz = grizli.utils.drizzle_array_groups(sci_list, wht_list, 
                     wcs_list, 
                     outputwcs=out_wcs, pixfrac=pixfrac, kernel=kernel,
                     verbose=False)
            
            drz_psf = _drz[0] / _drz[0].sum()               
            pyfits.writeto(f'irsa_{pixel_scale}pix_ch{ch}_{k}_psf.fits', data=drz_psf, overwrite=True)
        else:   
            if k == 0:
                _drz = grizli.utils.drizzle_array_groups(sci_list, wht_list, 
                         wcs_list, 
                         outputwcs=out_wcs, pixfrac=pixfrac, kernel=kernel,
                         verbose=False)
            else:
                _ = grizli.utils.drizzle_array_groups(sci_list, wht_list, 
                         wcs_list, 
                         outputwcs=out_wcs, pixfrac=pixfrac, kernel=kernel,
                         verbose=False, data=_drz[:3])
                     
    sci, wht, ctx, head, w = _drz  
    coswindow = CosineBellWindow(alpha=1)(_drz[0].shape)**0.05

    drz_psf = (_drz[0]*coswindow) / (_drz[0]*coswindow).sum()     
              
    if 0:
        pyfits.writeto(f'irsa_{pixel_scale}pix_ch{ch}_psf.fits', data=drz_psf, overwrite=True)
    else:
        return drz_psf, sci, wht
        

def recenter_psfs():
    
    import astropy.io.fits as pyfits
    pa = pyfits.open('cos-grism-j100012p0210-ch1-0.1.psfr_avg.fits') 
    pash = pa[0].data.shape
    xpa, ypa = np.indices(pash)
    xca = (pa[0].data*xpa).sum()/pa[0].data.sum()
    yca = (pa[0].data*ypa).sum()/pa[0].data.sum()
    
    pi = pyfits.open('/usr/local/share/python/golfir/golfir/data/psf/irsa_0.1pix_ch1_psf.fits')[0].data
    
    #pi = irac.warp_image(np.array([-2., -2]), pi)
    pi = np.roll(np.roll(pi, -2, axis=0), -2, axis=1)

    pish = pi.shape
    xpi, ypi = np.indices(pish)
    xci = (pi*xpi).sum()/pi.sum()
    yci = (pi*ypi).sum()/pi.sum()
    
    print(xca, yca, pash)
    print(xci, yci, pish)

def convolve_helper(data, kernel, method='fftconvolve', fill_scipy=False, cval=0.0):
    """
    Handle 2D convolution methods
    
    Parameters
    ==========
    
    method: str
        
        'fftconvolve':``scipy.signal.fftconvolve(data, kernel, mode='same')``

        'oaconvolve':``scipy.signal.oaconvolve(data, kernel, mode='same')``
    
        'stsci':``stsci.convolve.convolve2d(data, kernel, fft=1, mode='constant', cval=cval)``
        
        'xstsci': Try ``stsci`` but fall back to ``fftconvolve`` if failed to 
        `import stsci.convolve`.
     
    If ``fill_scipy=True`` or ``method='stsci'``, the ``data`` array will be 
    expanded to include the kernel size and padded with values given by 
    ``cval``.
    
    """
    
    if method == 'xstsci':
        try:
            from stsci.convolve import convolve2d
            method = 'stsci'
        except:
            print('import stsci.convolve failed.  Fall back to fftconvolve.')
            method = 'fftconvolve'
            
    if method in ['oaconvolve', 'fftconvolve']:
        from scipy.signal import fftconvolve, oaconvolve
        
        if method == 'fftconvolve':
            convolve_func = fftconvolve
        else:
            convolve_func = oaconvolve
        
        if fill_scipy:
            sh = data.shape
            shk = kernel.shape
            _data = np.zeros((sh[0]+2*shk[0], sh[1]+2*shk[1]))+cval
            _data[shk[0]:-shk[0], shk[1]:-shk[1]] = data
        else:
            _data = data
            
        conv = convolve_func(_data, kernel, mode='same')
        if fill_scipy:
            conv = conv[shk[0]:-shk[0], shk[1]:-shk[1]]
        
    elif method == 'stsci':
        from stsci.convolve import convolve2d
        conv = convolve2d(data, kernel, mode='constant', cval=cval, fft=1)
    
    else:
        raise ValueError("Valid options for `method` are 'fftconvolve',"
                         "'oaconvolve', 'stsci' ('xstsci').")
    
    return conv


def warp_image(transform, image, warp_args={'order': 3, 'mode': 'constant', 'cval': 0.0}, center=None):
    """
    Warp an image with `skimage.transform`
    
    Parameters
    ----------
    transform : array-like
        Transformation parameters
        
    image : 2D array-like
        Image to warp
    
    warp_args : dict
        Keyword arguments passed to `skimage.transform.warp`
    
    center : 2-element array or None
        Image center.  If `None`, calculated as ``image.shape/2.-1``
    
    Returns
    -------
    warped : array-like
        Shifted and rotated version of `image`
        
    """
    from skimage.transform import SimilarityTransform, warp, rotate

    trans = transform[0:2]
    if len(transform) > 2:
        rot = transform[2]
        if len(transform) > 3:
            scale = transform[3]
        else:
            scale = 1.
    else:
        rot = 0.
        scale = 1.

    if center is None:
        center = np.array(image.shape)/2.-1

    tf_rscale = SimilarityTransform(translation=trans/scale+center*(1-scale), rotation=None, scale=scale)

    shifted = warp(image, tf_rscale.inverse, **warp_args)
    rotated = rotate(shifted, rot, resize=False, center=center)

    return rotated


def argv_to_dict(argv, defaults={}, dot_dict=True):
    """
    Convert a list of (simple) command-line arguments to a dictionary.
    
    Parameters
    ----------
    argv : list of strings
        E.g., ``sys.argv[1:]``.
    
    defaults : dict
        Default dictionary
    
    dot_dict : bool
        If true, then intepret keywords with '.' as nested dictionary keys, 
        e.g., ``--d.key=val`` >> {'d': {'key': 'val'}}
        
    Examples:
    ---------
    
        # $ myfunc arg1 --p1=1 --l1=1,2,3 --pdict.k1=1 -flag
        >>> argv = 'arg1 --p1=1 --l1=1,2,3 --pdict.k1=1 -flag'.split()
        >>> args, kwargs = argv_to_dict(argv)
        >>> print(args)
        ['arg1']
        >>> print(kwargs)
        {'p1': 1, 'l1': [1, 2, 3], 'pdict': {'k1': 1}, 'flag': True}
        
        # With defaults
        defaults = {'pdict':{'k2':2.0}, 'p2':2.0}
        >>> args, kwargs = argv_to_dict(argv, defaults=defaults)
        >>> print(kwargs)
        {'pdict': {'k2': 2.0, 'k1': 1}, 'p2': 2.0, 'p1': 1, 'l1': [1, 2, 3], 'flag': True}
        
    """
    import copy
    import json
    
    kwargs = copy.deepcopy(defaults)
    args = []
    
    for i, arg in enumerate(argv):        
        if not arg.startswith('-'):
            # Arguments
            try:
                args.append(json.loads(arg))
            except:
                args.append(json.loads(f'"{arg}"'))
                
            continue
            
        spl = arg.strip('--').split('=')
        if len(spl) > 1:
            # Parameter values
            key, val = spl
            val = val.replace('True','true').replace('False','false')
            val = val.replace('None','null')
        else:
            # Parameters, set to true, e.g., -set_flag
            key, val = spl[0], 'true'
            
            # single -
            if key.startswith('-'):
                key = key[1:]
        
        # List values        
        if ',' in val:
            try:
                # Try parsing with JSON
                jval = json.loads(f'[{val}]')
            except:
                # Assume strings
                str_val = ','.join([f'"{v}"' for v in val.split(',')])
                jval = json.loads(f'[{str_val}]')
        else:
            try:
                jval = json.loads(val)
            except:
                # String
                jval = val
        
        # Dict keys, potentially nested        
        if dot_dict & ('.' in key):
            keys = key.split('.')
            d = kwargs
            for k in keys[:-1]:
                if k not in d:
                    d[k] = {}
                
                d = d[k]
                    
            d[keys[-1]] = jval
        else:            
            kwargs[key] = jval
            
    
    return args, kwargs    
    

def make_zodimodel():
    """
    Create precomputed model of the Zodiacal light seen by Spitzer
    
    `zodipy` requires Python 3.8
    
    """
    import astropy.time
    import astropy.units as u
    import zodipy
    import numpy as np
    import astropy.io.fits as pyfits
    
    # Few days after launch
    t0 = astropy.time.Time('2003-09-01')
    
    # Decomissioned
    t1 = astropy.time.Time('2020-01-28')
    
    # Time step, days
    tstep = 3
    times = np.arange(t0.mjd, t1.mjd, tstep)
    
    model = zodipy.InterplanetaryDustModel()
    
    nside = 32
    
    emission = model.get_instantaneous_emission(
        3.6*u.micron, 
        nside=nside, 
        observer="Spitzer", 
        epochs=times[0],  # 2010-01-01 (iso) in MJD
        coord_out="C"
    )
    
    sh = (len(times), emission.size)
    
    maps = {}
    for w in [3.6, 4.5, 5.8, 7.8]:
        maps[w] = np.zeros(sh, dtype=np.float32)
        
        for i, t in enumerate(times):
            print(i, t)
            maps[w][i,:] = model.get_instantaneous_emission(
                                        w*u.micron, 
                                        nside=nside, 
                                        observer="Spitzer", 
                                        epochs=t,
                                        coord_out="C"
                                    )
    
    h = pyfits.Header()
    h['NSIDE'] = nside, 'Healpix NSIDE'
    h['WAVE'] = w, 'Wavelength, microns'
    h['T0'] = t0.mjd, 'Start time, MJD'
    h['T1'] = t1.mjd, 'End type, MJD'
    h['TSTEP'] = tstep, 'Time step, days'
    
    for i, w in enumerate(maps):
        h['CHANNEL'] = i+1
        h['WAVE'] = w
        h['EXTNAME'] = f'{w:.1f}'
        if i == 0:
            hdu = pyfits.PrimaryHDU(data=maps[w], header=h)
            hdul = pyfits.HDUList([hdu])
        else:
            hdu = pyfits.ImageHDU(data=maps[w], header=h)
            hdul.append(hdu)
            
    hdul.writeto(f'spitzer_zodi_healpy_{nside}.fits', overwrite=True)


def get_zodi_file():
    """
    Get path to module zodi file
    """
    path = os.path.join(os.path.dirname(__file__), 
                        'data/spitzer_ch1_zodi_healpy_16.fits')
    return path


def get_spitzer_zodimodel(ra=150.1, dec=2.1, mjd=54090., zodi_file=None, ch=1, exptime=None, ch1_scale=4.):
    """
    Evaluate the zodi model at a given sky coordinate + epoch
    
    Parameters
    ----------
    ra : float
        Target Right Ascension, degrees
    
    dec : float
        Target declination, degrees
    
    mjd : float, array-like
        Modified Julian Date[s] of the observation
    
    zodi-file : str, `~astropy.io.fits.HDUList`
        Filename of the precomputed Spitzer zodi map, or the opened zodi_file.
        If `None`, then get the file distributed with the module in 
        `golfir.utils.get_zodi_file`.
    
    ch : int
        IRAC channel (1-4)
    
    exptime : float or array-like
        Exposure time used to calculate total noise
    
    Returns
    -------
    
    zodi_data : float, array-like
        Zodi data evaluated at the HEALPIX pixel of the target coordinates
        and interpolated at the specified times.  Returned in units of 
        MJy / steradian.
    
    noise_unit : float, array-like
        If `exptime` specified then return expected noise from sky + read
        noise in calibrated units MJy / steradian.
        
    """
    from astropy.coordinates import SkyCoord, FK5
    from astropy_healpix import HEALPix
    import astropy.units as u
    
    if zodi_file is None:
        zodi_file = get_zodi_file()
        
    if isinstance(zodi_file, str):
        zodi = pyfits.open(zodi_file)
    else:
        zodi = zodi_file
    
    if isinstance(ch, str):
        ch = int(ch[-1])
        
    # if len(zodi) > 1:
    #     ich = ch-1
    # else:
    #     ich = 0
    
    ich = np.minimum(len(zodi), ch)-1
    
    # Rouch scale corrections for single file with ch1 estimates
    if len(zodi) == 1:
        scale = {1:1, 2:3.5, 3:19, 4:73}[ch]
    else:
        scale = 1.
    
    scale *= ch1_scale
       
    h = zodi[ich].header
    nside = h['NSIDE']
    times = np.arange(h['T0'], h['T1'], h['TSTEP'])
    
    hp = HEALPix(nside=nside, order='ring', frame=FK5())
    coo = SkyCoord(ra, dec, unit=('deg','deg'))
    ix = hp.skycoord_to_healpix(coo)
    
    zodi_ix = zodi[ich].data[:,ix]
    zodi_data = np.interp(mjd, times, zodi_ix,
                          left=zodi_ix[0], right=zodi_ix[-1]) * scale
    
    # https://irsa.ipac.caltech.edu/data/SPITZER/docs/irac/iracinstrumenthandbook/62/
    RONOISE = [22.4, 23.7, 9.1, 7.1]
    GAIN = [3.7, 3.71, 3.8, 3.8]
    FLUXCONV = [0.1257, 0.1447, 0.5858, 0.2026]
    DARK = [0.05, 0.28, 1, 3.8]
    
    if exptime is not None:
        DN = zodi_data / FLUXCONV[ch-1]
        elec = GAIN[ch-1] * DN
        noise = np.sqrt((elec+DARK[ch-1])*exptime + RONOISE[ch-1]**2)/exptime
        noise_unit = noise / GAIN[ch-1] * FLUXCONV[ch-1]
    else:
        noise_unit = None
        
    return zodi_data, noise_unit
    
    
    