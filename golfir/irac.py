"""
Processing IRAC BCDs
"""

import glob
import os
import inspect
import traceback
from collections import OrderedDict

import numpy as np
import numpy.ma

import astropy.io.fits as pyfits
import astropy.wcs as pywcs
import astropy.units as u

try:    
    from photutils import (HanningWindow, TukeyWindow, 
                            CosineBellWindow, SplitCosineBellWindow, 
                            TopHatWindow)
except:
    from photutils.psf.matching import (HanningWindow, TukeyWindow, 
                            CosineBellWindow, SplitCosineBellWindow, 
                            TopHatWindow)

from .utils import warp_image, get_zodi_file, get_spitzer_zodimodel

try:
    from drizzlepac.astrodrizzle import ablot
except:
    print("(golfir.irac) Warning: failed to import drizzlepac")
    
try:
    from grizli import utils, prep
except:
    print("(golfir.irac) Warning: failed to import grizli")
    
def process_all(channel='ch1', output_root='irac', driz_scale=0.6, kernel='point', pixfrac=0.5, out_hdu=None, wcslist=None, pad=10, radec=None, aor_ids=None, use_brmsk=True, nmin=5, flat_background=False, two_pass=False, min_frametime=20, instrument='irac', run_alignment=True, assume_close=True, align_threshold=3, mips_ext='_bcd.fits', use_xbcd=False, ref_seg=None, save_state=True, med_max_size=500e6, global_mask=''):
    """
    """
    import glob
    import os
    import time
    
    import numpy as np
    import numpy.ma

    import astropy.io.fits as pyfits
    import astropy.wcs as pywcs
    import astropy.units as u

    from grizli import utils, prep
    
    if aor_ids is None:
        aor_ids = glob.glob('r[0-9]*[0-9]')
        aor_ids.sort()
        
    #aor_ids = ['r42421248']
    
    if instrument == 'irac':
        inst_key = 'SPITZER_I*cbcd.fits'
        output_label = channel
    else:
        inst_key = 'SPITZER_M*'+mips_ext
        output_label = channel.replace('ch', 'mips')
    
    if use_xbcd:
        inst_key = 'SPITZER*_xbcd.fits*'
            
    aors = OrderedDict()
    pop_list = []
    N = len(aor_ids)
    for i, aor in enumerate(aor_ids):
        files = glob.glob('{0}/{1}/bcd/{2}'.format(aor, channel, inst_key))
        # if len(files) > 0:
        #     print(aor, len(files))
        if len(files) < nmin:
            continue
        
        files.sort()
        
        print('\n#####\n{0}/{1}: {2} {3}\n#####\n'.format(i+1, N, aor, len(files)))
        
        aors[aor] = IracAOR(files=files, nskip=0, name=aor, channel=channel,
                            use_brmsk=use_brmsk, min_frametime=min_frametime,
                            instrument=instrument)
        
        if aors[aor].N == 0:
            pop_list.append(aor)
            
        if run_alignment: # & (instrument == 'irac'):
            try:                
                aors[aor].align_to_reference(reference=['GAIA','GAIA_Vizier'],
                                      radec=radec, threshold=align_threshold,
                                      assume_close=assume_close,
                                      med_max_size=med_max_size)
            except:
                # fp = open('{0}-{1}_wcs.failed'.format(aor, aors[aor].label),'w')
                # fp.write(time.ctime())
                # fp.close()
                failed = '{0}-{1}_wcs.failed.txt'.format(aor, aors[aor].label)
                utils.log_exception(failed, traceback)
                
                pop_list.append(aor)
        
    if len(aors) == 0:
        return {}
            
    for aor in pop_list:
        if aor in aors:
            p = aors.pop(aor)
    
    if len(aors) == 0:
        return {}
     
    # All wcs
    if wcslist is None:
        wcslist = []
        for aor in aors:
            wcslist.extend(aors[aor].wcs)
    
    #driz_scale = 0.5
    #kernel='point'
    
    if two_pass:
        niter = 2
    else:
        niter = 1
    
    for iter_pass in range(niter):
                    
        for i, aor in enumerate(aors):
            if iter_pass == 0:
                extra_mask = None
                force_pedestal = False
            else:
                extra_mask = aors[aor].extra_mask
                force_pedestal=True
                
            print(f'drizzle_simple (iter={iter_pass}): ', i, aor, aors[aor].N)
            aors[aor].drz = aors[aor].drizzle_simple(wcslist=wcslist, driz_scale=driz_scale, theta=0, kernel=kernel, out_hdu=out_hdu, pad=pad, pixfrac=pixfrac, flat_background=flat_background, med_max_size=med_max_size, force_pedestal=force_pedestal, extra_mask=extra_mask)#'point')
        
        if False:
            # Testing
            import grizli.ds9
            ds9 = grizli.ds9.DS9()
            
            i0 = int(ds9.get('frame'))
            for i, aor in enumerate(aor_ids):
                sci, wht, ctx, head, w = aors[aor].drz
                ds9.frame(i0+i+1)
                ds9.view(sci, header=head)
    
        for i, aor in enumerate(aors):
            sci, wht, ctx, head, w = aors[aor].drz
        
            root_i = '{0}-{1}-{2}'.format(output_root, aor, output_label)
            pyfits.writeto('{0}_drz_sci.fits'.format(root_i), data=sci, header=head, overwrite=True)
            pyfits.writeto('{0}_drz_wht.fits'.format(root_i), data=wht, header=head, overwrite=True)
        
            if i == 0:
                h0 = head.copy()
                h0['NDRIZIM'] = aors[aor].N - aors[aor].nskip
                h0['NAOR'] = 1
                num = sci*wht
                den = wht
            else:
                h0['EXPTIME'] += head['EXPTIME']
                h0['NDRIZIM'] += aors[aor].N - aors[aor].nskip
                h0['NAOR'] += 1
                num += sci*wht
                den += wht
        
            h0['AOR{0:05d}'.format(i)] = aor
            h0['PA_{0:05d}'.format(i)] = aors[aor].theta
            h0['N_{0:05d}'.format(i)] = aors[aor].N
            
        drz_sci = num/den
        drz_sci[den == 0] = 0
        drz_wht = den
    
        irac_root = '{0}-{1}'.format(output_root, output_label)
    
        pyfits.writeto('{0}_drz_sci.fits'.format(irac_root), data=drz_sci, header=h0, overwrite=True)
        pyfits.writeto('{0}_drz_wht.fits'.format(irac_root), data=drz_wht, header=h0, overwrite=True)
        
        if two_pass & (iter_pass == 0):
            print('#\n# Source mask for median background\n#\n')
            
            if ref_seg is None:
                bkg_params = {'bh': 16, 'bw': 16, 'fh': 3, 'fw': 3,
                               'pixel_scale': 0.5+1.5*(instrument == 'mips')}

                apertures = np.logspace(np.log10(0.3), np.log10(12), 5)*u.arcsec
                detect_params = prep.SEP_DETECT_PARAMS.copy()

                cat = prep.make_SEP_catalog(root=irac_root, threshold=1.5, 
                            get_background=True, bkg_params=bkg_params, 
                            phot_apertures=apertures, aper_segmask=True,
                            column_case=str.lower, 
                            detection_params=detect_params, 
                            pixel_scale=driz_scale)
            
                seg = pyfits.open('{0}_seg.fits'.format(irac_root))[0]
            else:
                seg = ref_seg
            
            seg_wcs = pywcs.WCS(seg.header)
            seg_wcs.pscale = utils.get_wcs_pscale(seg_wcs)
            if (not hasattr(seg_wcs, '_naxis1')) & hasattr(seg_wcs, '_naxis'):
                seg_wcs._naxis1, seg_wcs._naxis2 = seg_wcs._naxis
            
            #mask_file = '{0}_mask.reg'.format(output_root)
            if os.path.exists(global_mask):
                import pyregion
                print('Additional mask from {0}'.format(global_mask))
                mask_reg = pyregion.open(global_mask)
                mask_im = mask_reg.get_mask(hdu=seg)
                seg.data += mask_im*1
                
            for i, aor in enumerate(aors):
                print('\n {0} {1} \n'.format(i, aor))
                blot_seg = aors[aor].blot(seg_wcs, seg.data, interp='poly5')
                med = aors[aor].med2d[0]-aors[aor].pedestal[0]
                aors[aor].extra_mask = (blot_seg > 0) #| (med < -0.05)
                
    if save_state:
        for i, aor in enumerate(aors):
            aors[aor].save_state(mips_ext=mips_ext, verbose=True)
            aors[aor].tab = aors[aor].wcs_table()
        
    return aors


dx = 0.5
dp = [0,0]

def get_bcd_psf(ra=0, dec=0, wcs=None, channel=1, dx=dx, dd=dp):
    """
    Get the (Warm mission) psf at a given position

    618  ls PSF/Warm/IRACPC1_*fits | sed "s/_/ /g" | awk '{print $2,$3}' | sed "s/col//" | sed "s/row//" | sed "s/.fits//" > PSF/Warm/ch1.xy
    619  ls PSF/Warm/IRACPC2_*fits | sed "s/_/ /g" | awk '{print $2,$3}' | sed "s/col//" | sed "s/row//" | sed "s/.fits//" > PSF/Warm/ch2.xy

    """
    import numpy as np
    import numpy.ma

    import astropy.io.fits as pyfits
    import astropy.wcs as pywcs

    xy = np.array(wcs.all_world2pix([ra], [dec], 0)).flatten()
        
    if (xy.min() < -12) | (xy.max() > 256+12):
        return np.zeros((256,256))
        
    psf_xy = np.loadtxt('PSF/Warm/ch{0}.xy'.format(channel), dtype=int)
    dr = np.sqrt((xy[0]-psf_xy[:,0])**2+(xy[1]-psf_xy[:,1])**2)
    ix = np.argmin(dr)
    
    psf_im = pyfits.open('PSF/Warm/IRACPC{0}_col{1:03d}_row{2:03d}.fits'.format(channel, psf_xy[ix,0], psf_xy[ix,1]))[0].data
    
    phase = np.cast[int]((xy+dx-np.floor(xy+dx))*5)
    print(xy, phase)
    
    if dp != [0,0]:
        psf_im = np.roll(np.roll(psf_im, dp[0], axis=0), dp[1], axis=1)
        
    psf_x = psf_im[phase[1]:125:5, phase[0]:125:5]
    
    pad = 100
    xp = np.cast[int](np.floor(xy))+pad
    psf_full = np.zeros((256+2*pad,256+2*pad))
    psf_full[xp[1]-12:xp[1]+13, xp[0]-12:xp[0]+13] += psf_x
    return psf_full[pad:-pad, pad:-pad]
         
GAIA_SIZE = 10.
                             
class IracAOR():
    def __init__(self, files=[], nskip=2, name='empty', channel='ch1', use_brmsk=True, min_frametime=20, instrument='irac'):
        
        self.name = name
        self.channel = channel
        
        self.N = len(files)
        self.files = files
        self.files.sort()
        self.nskip = nskip
        self.instrument = instrument
        
        if instrument == 'irac':
            self.label = self.channel
            self.native_scale = 1.223
        else:
            self.label = self.channel.replace('ch', instrument)
            self.native_scale = 2.50
            
        self.read_files(use_brmsk=use_brmsk, min_frametime=min_frametime)
    
    @staticmethod
    def read_cbcd(file, instrument='irac', use_brmsk=True):
        
        # if 'xbcd' in file:
        #     _res = read_xbcd(file)
        #     return _res
        
        with pyfits.open(file) as im:
            cbcd = im[0].data.astype(np.float32)
            wcs = pywcs.WCS(im[0].header)
            wcs.pscale = utils.get_wcs_pscale(wcs)
        
        with pyfits.open(file.replace('bcd.', 'bunc.')) as hdul:
            cbunc = hdul[0].data.astype(np.float32)
        
        if instrument == 'irac':
            bimsk_file = file.replace('_cbcd','_bcd')
            bimsk_file = bimsk_file.replace('bcd.', 'bimsk.')
        else:
            bimsk_file = file.replace('_cbcd','_bcd')
            bimsk_file = bimsk_file.replace('bcd.', 'bbmsk.')
            bimsk_file = bimsk_file.replace('_ebb','_eb')
        
        with pyfits.open(bimsk_file) as hdul:
            bimsk = hdul[0].data.astype(np.float32)
        
        brmsk = np.zeros(bimsk.shape, dtype=np.int16)
        if use_brmsk:
            rmsk_file =  file.replace('bcd.', 'brmsk.')
            if os.path.exists(rmsk_file):
                with pyfits.open(rmsk_file) as hdul:
                    brmsk = hdul[0].data.astype(np.int16)
                            
        return None, cbcd, cbunc, bimsk, brmsk, wcs
        
    @staticmethod
    def read_xbcd(file):
        with pyfits.open(file) as im:
            cbcd = im['CBCD'].data.astype(np.float32)
            cbunc = im['CBUNC'].data.astype(np.float32)
            bimsk = im['WCS'].data.astype(np.int16)
        
            wcs = pywcs.WCS(im['WCS'].header)
            wcs.pscale = utils.get_wcs_pscale(wcs)
        
        return None, cbcd, cbunc, bimsk, bimsk*1, wcs
        
    def read_files(self, use_brmsk=True, min_frametime=20):
        
        self.hdu = [pyfits.open(file) for file in self.files]
        for i in  np.arange(len(self.hdu))[::-1]:
            if self.hdu[i][0].header['EXPTIME'] < min_frametime:
                self.hdu.pop(i)
                self.files.pop(i)
        
        self.nskip = 0
        self.N = len(self.files)
        if self.N == 0:
            return None
        
        self.cbcd = []
        self.cbunc = []
        self.bimsk = []
        self.brmsk = []
        self.wcs = []
        
        for file in self.files:
            if '_xbcd' in file:
                _res = self.read_xbcd(file)
            else:
                _res = self.read_cbcd(file, use_brmsk=True, 
                                      instrument=self.instrument)   
            
            self.cbcd.append(_res[1])
            self.cbunc.append(_res[2])
            self.bimsk.append(_res[3])
            self.brmsk.append(_res[4])
            self.wcs.append(_res[5])
            
        self.cbcd = np.stack(self.cbcd)
        self.cbunc = np.stack(self.cbunc)
        self.bimsk = np.stack(self.bimsk)
        self.brmsk = np.stack(self.brmsk)
        
        self.shifts = np.zeros((self.N, 2))
        self.rot = np.zeros(self.N)
        
        self.shift_wcs = [utils.transform_wcs(self.wcs[i], translation=self.shifts[i,:], rotation=self.rot[i], scale=1.0) for i in range(self.N)]
         
        self.dq = (~np.isfinite(self.cbcd)) | (~np.isfinite(self.cbunc))
        self.dq |= self.bimsk > 0
        
        if self.brmsk is not None:
            self.dq |= self.brmsk > 0
            
        self.orig_dq = self.dq
        
        self.cbcd[self.dq] = 0
        self.cbunc[self.dq] = 0
        self.err_scale = 1.
        
        #self.wht = np.ones_like(self.cbcd)
        self.ivar = 1/self.cbunc**2
        self.ivar[~np.isfinite(self.ivar)] = 0
        
        self.rescale_uncertainties()
        
    def get_median(self, extra_mask=None):
        
        if extra_mask is not None:
            ma = np.ma.masked_array(self.cbcd, mask=self.dq | extra_mask)
        else:
            ma = np.ma.masked_array(self.cbcd, mask=self.dq)
        
        pedestal = np.array([np.ma.median(ma[i,:,:]) for i in range(self.N)])
        global_med = np.ma.median((ma.T-pedestal).T[self.nskip:,:,:],axis=0)
        
        by_image = np.stack([pedestal[i]+global_med for i in range(self.N)])
        
        return pedestal, by_image.data
    
    def rescale_uncertainties(self):
        pedestal, med2d = self.get_median()
        m = (self.cbunc > 0) & (self.dq == 0)
        res = ((self.cbcd.T-pedestal).T/self.cbunc)[m]
        nmad = utils.nmad(res)
        if not np.isfinite(nmad):
            nmad = 1.
            
        self.err_scale = nmad
        self.ivar = 1/(self.cbunc*self.err_scale)**2
        self.ivar[~np.isfinite(self.ivar)] = 0
        print('err_scale: {0:.2f}'.format(self.err_scale))
        
    def get_nmad(self, median=0):
        ma = np.ma.masked_array(self.cbcd, mask=self.dq)
        
        return 1.48*np.ma.median(np.ma.abs(ma-median)[self.nskip:,:,:], axis=0).data
    
    @property
    def theta(self):
        """
        Position angle calculated from CD matrix
        """
        if len(self.wcs) == 0:
            return 0
        else:
            cd = self.wcs[0].wcs.cd
            theta = np.arctan2(cd[1][0], cd[1][1])/np.pi*180
            return theta
            
    def get_theta(self):
        px = self.wcs[0].calc_footprint()
        dx = np.diff(px, axis=0)
        theta = np.arctan(dx[0,0]/dx[0,1])
        return theta
    
    def make_median_image(self, pixel_scale=None, extra_mask=None, sigma_clip=20, max_size=50e6):
        
        if pixel_scale is None:
            pixel_scale = self.wcs[0].pscale
        
        pedestal, med2d = self.get_median(extra_mask=extra_mask)
        
        med_hdu = utils.make_maximal_wcs(self.wcs, pixel_scale=pixel_scale, theta=-self.theta, pad=10, get_hdu=True, verbose=False)
        sh = med_hdu.data.shape
        med_wcs = pywcs.WCS(med_hdu.header)
        med_wcs.pscale = pixel_scale
        
        NPIX = self.N*sh[0]*sh[1]
        if NPIX > max_size:
            print('NPIX = {0:.1f}M, use sigma-clipped'.format(NPIX/1e6))
            
            # Sigma-clipped drizzle
            for i in range(self.N):
                drz = utils.drizzle_array_groups([(self.cbcd-med2d)[i,:,:]], [(self.ivar*(self.dq == 0))[i,:,:]], [self.shift_wcs[i]], outputwcs=med_wcs, kernel='point', verbose=False)
                
                if i == 0:
                    med_data = drz[0]*1
                    med_n = (drz[1] > 0)*1
                    continue
                    
                # CR in new image
                keep = (drz[1] > 0) & ((drz[0] - med_data)*np.sqrt(drz[1]) < sigma_clip)
                keep |= (med_n == 0) & (drz[1] > 0)
                
                med_data = med_data*med_n + drz[0]*keep
                med_n += keep*1
                med_data /= np.maximum(med_n, 1)
            
            return med_wcs, med_data
                
        else:
            med_data = np.zeros((self.N, sh[0], sh[1]), dtype=np.float32)
        
            for i in range(self.N):
                drz = utils.drizzle_array_groups([(self.cbcd-med2d)[i,:,:]], [(self.ivar*(self.dq == 0))[i,:,:]], [self.shift_wcs[i]], outputwcs=med_wcs, kernel='point', verbose=False)
                med_data[i,:,:] = drz[0]
        
            med_ma = np.ma.masked_array(med_data, mask=med_data == 0)
         
            med = np.ma.median(med_ma[self.nskip:,:,:], axis=0)
        
            return med_wcs, med.data
        
    def blot(self, med_wcs, med, interp='poly5'):
    
        blot_data = np.zeros(self.cbcd.shape, dtype=np.float32)
        for i in range(self.N):
            blot_data[i,:,:] = ablot.do_blot(med.astype(np.float32), med_wcs, self.shift_wcs[i], 1, coeffs=True, interp=interp, sinscl=1.0, stepsize=10, wcsmap=None)
    
        return blot_data
        
    def get_shifts(self, threshold=0.8, ref=None, use_triangles=False):
        import astropy.units as u
        from grizli import prep
        
        pedestal, med2d = self.get_median()
        nmad = self.get_nmad(median=med2d)
        
        detection_params={'clean': True, 'filter_type': 'conv', 'deblend_cont': 0.005, 'deblend_nthresh': 32, 'clean_param': 1, 'minarea': 16, 'filter_kernel': None}
             
        cats, segs = [0]*self.nskip, [0]*self.nskip
        for i in range(self.nskip, self.N):
            simple = np.median(self.cbcd[i,:,:][self.dq[i,:,:] == 0])
            cat, seg = prep.make_SEP_catalog_from_arrays(self.cbcd[i,:,:]-simple, self.cbunc[i,:,:], (self.dq[i,:,:] > 0), segmentation_map=True, threshold=threshold, wcs=self.shift_wcs[i], get_background=True)
            
            cat['mag'] = 23.9-2.5*np.log10(cat['flux'])
            
            cats.append(cat)
            segs.append(seg)
        
        if ref is None:
            ref = cats[-1]

        s0 = self.shifts*1.
        
        for i in range(self.nskip, self.N):
            idx, dr = ref.match_to_catalog_sky(cats[i])
            #plt.hist(dr, bins=100, range=[0,10], alpha=0.5)
            ok = (dr.value < 0.8) & (cats[i]['peak']/cats[i]['flux'] < 0.5)
            
            if use_triangles:
                NR = 80
                ok &= np.random.rand(len(ok)) < NR/len(ok)

                tri = cats[i][ok].match_triangles(ref[idx][ok], self_wcs=self.shift_wcs[i], x_column='x', y_column='y', mag_column='mag', other_ra='ra', other_dec='dec', pixel_index=1, match_kwargs={'ignore_scale':True, 'ignore_rot':True, 'size_limit':[20, 100]}, pad=1, show_diagnostic=0, auto_keep=7, maxKeep=10, auto_limit=3, ba_max=0.8, scale_density=10000)
                        
                self.shifts[i,:] += tri[1].translation
                self.rot[i] += tri[1].rotation
            else:
                xp, yp = np.array(self.shift_wcs[i].all_world2pix(ref['ra'][idx], ref['dec'][idx], 0)).T
                dx = np.median((cats[i]['x']-xp)[ok])
                dy = np.median((cats[i]['y']-yp)[ok])
                self.shifts[i,:] -= np.array([dx, dy])
                
        ds = self.shifts - s0
        
        return ds


    def drizzle_simple(self, force_pedestal=True, dq_sn=5, fmax=0.5, flat_background=False, extra_mask=None, med_max_size=60e6, **kwargs):
        
        if (not hasattr(self, 'pedestal')) | force_pedestal: 
            med_wcs, med = self.make_median_image(max_size=med_max_size)        
            blot_data = self.blot(med_wcs, med)        
            pedestal, med2d = self.get_median(extra_mask=extra_mask)
        
            self.pedestal = pedestal
            self.med2d = med2d
            self.blot_data = blot_data
            self.med_wcs = med_wcs
            self.med = med
        
            # Add extra term in bright sources
            #fmax = 0.5
            eblot = 1-np.clip(blot_data, 0, fmax)/fmax
            self.dq = self.orig_dq |  ((self.cbcd - med2d - blot_data)*np.sqrt(self.ivar)*eblot > dq_sn) | ((self.cbcd - med2d - blot_data)*np.sqrt(self.ivar)*eblot < -dq_sn)
        
        if flat_background:
            bkg = self.pedestal
        else:
            bkg = self.med2d.T
            
        res = self.drizzle_output(pedestal=bkg, **kwargs)
        
        sci, wht, ctx, head, w = res
        wht *= (w.pscale/self.native_scale)**-4
        
        return sci, wht, ctx, head, w
        
        # Both
        # num = res[0]*res[1]+r0[0]*r0[1]
        # den = (res[1]+r0[1])
        # 
        # sci = num/den
        # sci[den == 0] = 0
        # wht = den
        # 
        # r1 = s1.drizzle_output(driz_scale=0.6, kernel='point', theta=0, wcslist=wcslist)
    
    @property
    def ABZP(self):
        """
        Convert MJy/Sr to an AB zeropoint for a given pixel scale
        """
        import astropy.units as u
        
        inp = u.MJy/u.sr
        out = u.uJy/u.arcsec**2
        ZP = 23.9 - 2.5*np.log10(((1*inp).to(out)*(self.native_scale*u.arcsec)**2).value)
        return ZP
    
    @property
    def EXPTIME(self):
        """
        Convert MJy/Sr to an AB zeropoint for a given pixel scale
        """
        return np.sum([self.hdu[i][0].header['EXPTIME'] for i in range(self.nskip, self.N)])
    
    def get_all_psfs(self, psf_coords=None, func=get_bcd_psf):
        
        ra, dec = psf_coords
        full_psf = np.zeros((self.N, 256, 256), dtype=np.float32)
        channel = int(self.channel[-1])
        for i in range(self.N):
            full_psf[i,:,:] = func(ra=ra, dec=dec, wcs=self.wcs[i],
                                          channel=channel) 
        return full_psf
        
    def drizzle_output(self, driz_scale=None, pixfrac=0.5, kernel='square', theta=None, wcslist=None, out_hdu=None, pedestal=0, pad=10, psf_coords=None):

        if driz_scale is None:
            driz_scale = self.wcs[0].pscale
        
        if theta is None:
            theta = -self.theta
        
        if wcslist is None:
            wcslist = self.wcs
                        
        if out_hdu is None:
            out_hdu = utils.make_maximal_wcs(wcslist, pixel_scale=driz_scale, theta=theta, pad=pad, get_hdu=True, verbose=False)
        
        sh = out_hdu.data.shape
        out_wcs = pywcs.WCS(out_hdu.header)
        out_wcs.pscale = utils.get_wcs_pscale(out_wcs)

        #pedestal, med2d = self.get_median()
        if psf_coords is None:
            in_sci = (self.cbcd.T-pedestal).T[self.nskip:,:,:]
        else:
            if len(psf_coords) == 3:
                psf_model = self.get_all_psfs(psf_coords=psf_coords[:2], func=psf_coords[2])
            else:
                psf_model = self.get_all_psfs(psf_coords=psf_coords)
            in_sci = psf_model[self.nskip:,:,:]
            
        in_wht = (self.ivar*(self.dq == 0))[self.nskip:,:,:]
        
        sci, wht, ctx, head, w = utils.drizzle_array_groups(in_sci, in_wht, 
                           self.shift_wcs[self.nskip:], outputwcs=out_wcs, 
                           kernel=kernel, pixfrac=pixfrac, verbose=False)
        
        head['EXPTIME'] = (self.EXPTIME, 'Total exposure time')
        
        # Flux units
        un = 1*u.MJy/u.sr
        #to_ujy_px = un.to(u.uJy/u.arcsec**2).value*(out_wcs.pscale**2)
        to_ujy_px = un.to(u.uJy/u.arcsec**2).value*(self.native_scale**2)

        #head['ABZP'] = (23.9, 'AB zeropoint')
        #head['PHOTFNU'] = 10**(-0.4*(head['ABZP']-8.90))
        
        head['ABZP'] = (23.9, 'AB zeropoint (uJy)')
        head['PHOTFNU'] = (1.e-6, 'Flux conversion to Jy')
        head['ORIGPIX'] = (self.native_scale, 'Assumed native pixel scale')
        
        head['FSCALE'] = (to_ujy_px, 'Flux scaling from cbcd units (MJy/sr)')
        head['BUNIT'] = 'microJy'
        
        sci *= to_ujy_px
        wht /= to_ujy_px**2
        
        head['INSTRUME'] = self.instrument.upper()
        head['FILTER'] = self.label.upper()
        
        return sci, wht, ctx, head, w
    
    def align_to_reference(self, reference=['GAIA','GAIA_Vizier','PS1'], radec=None, kernel='square', pixfrac=0.8, threshold=3, assume_close=True, med_max_size=60e6, **kwargs):
        from grizli import prep
        import astropy.units as u
        from drizzlepac import updatehdr
        
        res = self.drizzle_simple(dq_sn=5, fmax=0.5, driz_scale=1.0, kernel=kernel, pixfrac=pixfrac, theta=self.theta, med_max_size=med_max_size)
        sci, wht, ctx, head, w = res
        
        pyfits.writeto('{0}-{1}_drz_sci.fits'.format(self.name, self.label), data=sci, overwrite=True, header=head)

        pyfits.writeto('{0}-{1}_drz_wht.fits'.format(self.name, self.label), data=wht, overwrite=True, header=head)
        
        bkg_params = {'bh': 16, 'bw': 16, 'fh': 3, 'fw': 3, 'pixel_scale': 0.5+1.5*(self.instrument == 'mips')}
        
        cat = prep.make_SEP_catalog(root='{0}-{1}'.format(self.name, self.label), threshold=threshold, get_background=True, bkg_only=False, bkg_params=bkg_params, verbose=True, sci=None, wht=None, phot_apertures='6, 8.335, 16.337, 20', rescale_weight=True, column_case=str.upper, save_to_fits=True, source_xy=None, autoparams=[2.5, 3.5], mask_kron=False, max_total_corr=2, err_scale=1.)
        
        if self.instrument != 'mips':
            clip = (cat['MAG_AUTO'] > 15) & (cat['MAG_AUTO'] < 20) & (cat['FLUX_RADIUS'] > 1) & (cat['FLUX_RADIUS'] < 2)
        else:
            clip = np.isfinite(cat['MAG_AUTO'])
            cat['MAGERR_AUTO'] = 0.01
            
        cat[clip].write('{0}-{1}.cat.fits'.format(self.name, self.label), overwrite=True)
        
        ra, dec = w.wcs.crval
        radec_ext, ref_name = prep.get_radec_catalog(ra=ra, dec=dec, radius=GAIA_SIZE, product='{0}-{1}'.format(self.name, self.label), verbose=True, reference_catalogs=reference, use_self_catalog=False, date=self.hdu[0][0].header['MJD_OBS'], date_format='mjd')
        if radec is None:
            radec = radec_ext
            if assume_close > 0:
                ref = utils.read_catalog(radec_ext)
                idx, dr = ref.match_to_catalog_sky(cat)
                clip = dr.value < assume_close*1
                prep.table_to_regions(ref[idx][clip], radec_ext.replace('.radec','.reg'))  
                prep.table_to_radec(ref[idx][clip], radec_ext)  
        else:
            ref_name = 'USER'
        
        if self.instrument == 'mips':
            rd = utils.read_catalog(radec)
            idx, dr = rd.match_to_catalog_sky(cat[clip])
            keep = dr.value < assume_close*25
            radec = '{0}_tmp.radec'.format(self.name)
            prep.table_to_radec(rd[idx][keep], radec)
            
        if self.instrument == 'mips':
            mag_limits = [10, 30]
        else:
            mag_limits = [14,25]
            
        ali = prep.align_drizzled_image(root='{0}-{1}'.format(self.name, self.label), mag_limits=mag_limits, radec=radec, NITER=3, clip=20, log=True, outlier_threshold=5, verbose=True, guess=[0.0, 0.0, 0.0, 1], simple=True, rms_limit=2, use_guess=False, triangle_size_limit=[5, 1800], triangle_ba_max=0.9, max_err_percentile=99)
        
        orig_wcs, drz_wcs, out_shift, out_rot, out_scale = ali
        
        self.alignment = out_shift, out_rot, out_scale
        
        for i, h in enumerate(self.hdu):
            print(i, self.files[i])
            tmpfile = '{0}_{1}_tmp.fits'.format(self.name, self.label)
            h[0].writeto(tmpfile, overwrite=True)
            updatehdr.updatewcs_with_shift(tmpfile, 
                        '{0}-{1}_drz_sci.fits'.format(self.name, self.label),
                                      xsh=out_shift[0], ysh=out_shift[1],
                                      rot=out_rot, scale=out_scale,
                                      wcsname=ref_name, force=True,
                                      reusename=True, verbose=False,
                                      sciext='PRIMARY')
            
            self.hdu[i] = pyfits.open(tmpfile)
        
        self.wcs = [pywcs.WCS(im[0].header) for im in self.hdu]
        for w in self.wcs:
            w.pscale = utils.get_wcs_pscale(w)
        
        self.shifts = np.zeros((self.N, 2))
        self.rot = np.zeros(self.N)
        
        self.shift_wcs = [utils.transform_wcs(self.wcs[i], translation=self.shifts[i,:], rotation=self.rot[i], scale=1.0) for i in range(self.N)]
    
    def wcs_table(self):
        """
        Make a table of the wcs parameters of the component bcd files
        """
        
        tab = utils.GTable()
        tab['file'] = self.files
        tab['aor'] = [f.split('/')[0] for f in self.files]
        
        for k in ['EXPTIME', 'MJD_OBS', 'PROGID','DPID','OBJECT','OBSRVR']:
            tab[k.lower()] = np.array([im[0].header[k] for im in self.hdu])
        
        tab.rename_column('dpid', 'bcd')
            
        tab['crpix'] = np.array([w.wcs.crpix for w in self.wcs])
        tab['crval'] = np.array([w.wcs.crval for w in self.wcs])
        tab['cd'] = np.array([w.wcs.cd for w in self.wcs])
        tab['corners'] = np.array([w.calc_footprint() for w in self.wcs])
        tab['theta'] = np.array([np.arctan2(cd[1][0], cd[1][1])/np.pi*180 for cd in tab['cd']])
        
        tab.write('{0}-{1}.log.fits'.format(self.name, self.label), overwrite=True)
        return tab
           
    def save_state(self, mips_ext='_bcd.fits', ext='_wcs.fits', verbose=True):
        from grizli import utils
        
        if hasattr(self, 'pedestal'):
            bkg = self.pedestal
        else:
            bkg = np.zeos(self.N)
        
        for i in range(self.N):
            #dq_i = self.dq[i,:,:]
            head = utils.to_header(self.wcs[i])
            head['PEDESTAL'] = (bkg[i], 'Pedestal background value')
            
            if hasattr(self, 'alignment'):
                head['ASHIFTX'] = (self.alignment[0][0], 'Alignment x shift')
                head['ASHIFTY'] = (self.alignment[0][1], 'Alignment y shift')
                head['AROT'] = (self.alignment[1], 'Alignment rot')
                head['ASCALE'] = (self.alignment[2], 'Alignment scale')

            if self.instrument == 'irac':
                outfile = self.files[i].replace('_cbcd.fits', ext)
            else:
                mips_suffix = '_' + self.files[i].split('_')[-1]
                outfile = self.files[i].replace(mips_suffix, ext)
            
            if verbose:
                print(outfile)
            
            pyfits.writeto(outfile, data=self.dq[i,:,:]*1, header=head,
                           overwrite=True, output_verify='fix')
        
        # 2D background
        if hasattr(self, 'med2d'):
            if verbose:
                print('{0}-{1}_med.fits'.format(self.name, self.label))
                
            pyfits.writeto('{0}-{1}_med.fits'.format(self.name, self.label), 
                           data=self.med2d[0,:,:] - self.pedestal[0],
                           overwrite=True)


def mosaic_psf(output_root='irac', channel=1, pix=0.5, target_pix=0.1, pixfrac=0.2, kernel='square', aors={}, size=18, subtract_background=True, segmentation_mask=False, theta=0, native_orientation=False, instrument='irac', max_R=3.5, ds9=None):
    from grizli import prep
    from astropy.modeling.models import Gaussian2D  
    import astropy.units as u
    from skimage.morphology import binary_dilation
    
    import matplotlib.pyplot as plt
    
    from astropy.visualization import LogStretch, ImageNormalize, ManualInterval
                                           
    detect_params = prep.SEP_DETECT_PARAMS.copy()
    
    # Kernel
    fwhm = [1.95, 2.02, 1.9, 2.0][channel-1]
    r = fwhm/pix/2.35
    nr = int(np.round(np.log(2*5*r+1)/np.log(2)))
    yp, xp = np.indices((2**nr, 2**nr))
    kern = Gaussian2D(x_mean=2**(nr-1)-0.5, y_mean=2**(nr-1)-0.5,
                      x_stddev=r, y_stddev=r, amplitude=1)(xp, yp)
    detect_params['filter_kernel'] = kern
    
    bkg_params = {'bh': 16, 'bw': 16, 'fh': 3, 'fw': 3, 'pixel_scale': 0.5}
    apertures = np.logspace(np.log10(0.3), np.log10(12), 20)*u.arcsec
    
    if instrument == 'irac':
        label = 'ch{0}'.format(channel)
    else:
        label = 'mips{0}'.format(channel)
        
    ch_root='{0}-{1}'.format(output_root, label)
    cat = prep.make_SEP_catalog(root=ch_root, threshold=1.8, 
                                get_background=True, bkg_params=bkg_params, 
                                phot_apertures=apertures, aper_segmask=True,
                                column_case=str.lower, 
                                detection_params=detect_params, 
                                pixel_scale=pix)
    
    seg = pyfits.open('{0}_seg.fits'.format(ch_root))[0]
    seg_data = seg.data
    seg_wcs = pywcs.WCS(seg.header)
    
    for aor in aors:
        
        ### GAIA positions
        gaia_file = glob.glob('{0}-{1}_gaia*.radec'.format(aor, label))[0]
        gaia = utils.read_catalog(gaia_file)
        idx, dr = gaia.match_to_catalog_sky(cat)
        has_gaia = dr < 1*u.arcsec
        dd = gaia['dec'][idx] - cat['dec']
        dr = gaia['ra'][idx] - cat['ra']
        has_gaia &= np.isfinite(dd) & (np.isfinite(dr))
        
        cat['rax'] = cat['ra'] + np.median(dr[has_gaia])
        cat['decx'] = cat['dec'] + np.median(dd[has_gaia])
        
        clip = np.isfinite(cat['rax']+cat['decx'])
        cat = cat[clip]
        
        idx, dr = gaia.match_to_catalog_sky(cat, other_radec=('rax', 'decx'))
        has_gaia = (dr < 0.4*u.arcsec) & (cat['mag_auto'] > 16) & (cat['mag_auto'] < 20) & (cat['mask_aper_10'] < 3) & (cat['flux_radius'] < 2.5*0.5/pix)

        ### Normalization and background
        flux_scale = cat['flux_aper_10']*1
        aper_area = np.pi*(cat.meta['aper_10'][0]/2)**2 - cat['mask_aper_10']
        if subtract_background:
            bkg_pix = cat['bkg_aper_10']/aper_area*(target_pix/pix)**2
        else:
            bkg_pix = flux_scale*0.
            
        has_gaia &= np.isfinite(flux_scale) & np.isfinite(bkg_pix)
        flux_scale = flux_scale[has_gaia]
        bkg_pix = bkg_pix[has_gaia]
        ids = cat['number'][has_gaia]
        
        if native_orientation:
            cd = aors[aor].wcs[0].wcs.cd
                        
            #theta = -180+np.arctan2(cd[0][0], cd[0][1])/np.pi*180
            #theta = 90 + np.arctan2(cd[1][0], cd[1][1])/np.pi*180
            theta = np.arctan2(cd[1][0], cd[1][1])/np.pi*180
            
        sci_sum = None
        # Drizzle stars
        nstars = has_gaia.sum()
        gra, gdec = gaia['ra'][idx][has_gaia], gaia['dec'][idx][has_gaia]
        
        hdu = utils.make_wcsheader(0, 0, size=size, pixscale=target_pix,
                                   theta=-theta, get_hdu=True)
                
        # hdu.header['CRPIX1'] += 0.5
        # hdu.header['CRPIX2'] += 0.5
        
        yp, xp = np.indices(hdu.data.shape)
        R = np.sqrt((xp-size/2/target_pix)**2+(yp-size/2/target_pix)**2)*target_pix
        
        for i, (sra, sdec, scl, sbkg, sid) in enumerate(zip(gra, gdec, flux_scale, bkg_pix, ids)):
            hdu.header['CRVAL1'] = sra
            hdu.header['CRVAL2'] = sdec
            
            _res = aors[aor].drizzle_output(pedestal=aors[aor].pedestal,
                                            wcslist=None, 
                                            driz_scale=target_pix, 
                                            theta=theta, kernel=kernel, 
                                            out_hdu=hdu, pad=0, 
                                            pixfrac=pixfrac)
            
            sci_i, wht_i, ctx, head, w = _res
            #msk = (~np.isfinite(sci_i)) | (~np.isfinite(wht_i))
            
            sci_i -= sbkg
            # Blot segmentation
            if segmentation_mask:
                xy_seg = np.cast[int](np.round(seg_wcs.all_world2pix([sra], [sdec], 0))).flatten()
                n_seg = int(size/pix/2*1.5)
                slx = slice(xy_seg[0]-n_seg, xy_seg[0]+n_seg)
                sly = slice(xy_seg[1]-n_seg, xy_seg[1]+n_seg)
            
                blot_seg = utils.blot_nearest_exact(seg_data[sly, slx], seg_wcs.slice((sly, slx)), w, verbose=True, stepsize=10, scale_by_pixel_area=False)
                mask = (blot_seg > 0) & (blot_seg != sid)
                mask = ~mask
                
            else:
                mask = (R > max_R) & (sci_i/scl*1000*(0.1/target_pix)**2 > 0.1)
                nb=int(np.round(17*0.1/target_pix))
                mask = 1-binary_dilation(mask, selem=np.ones((nb,nb)))
                
                
            if sci_sum is None:
                wht_sum = wht_i*scl**2*mask
                sci_sum = sci_i*wht_i*scl*mask
            else:
                wht_sum += wht_i*scl**2*mask
                sci_sum += sci_i*wht_i*scl*mask

            print(i, sra, sdec, scl, (1-mask).sum(), wht_i.max())

            if ds9 is not None:
                ds9.view(sci_sum/wht_sum*1000)
        
        if sci_sum is None:
            continue
                             
        psf_sci = sci_sum/wht_sum
        psf_sci[wht_sum == 0] = 0
        psf_wht = wht_sum*psf_sci.sum()**2
        psf_sci /= psf_sci.sum()
        aors[aor].psf_sci = psf_sci
        aors[aor].psf_wht = psf_wht
        
        fig = plt.figure(figsize=[4,4])
        ax = fig.add_subplot(111)
        interval = ManualInterval(vmin=-0.*target_pix**2, 
                                  vmax=1*target_pix**2)
        norm = ImageNormalize(psf_sci, interval=interval,
                              stretch=LogStretch())
                              
        ax.imshow(psf_sci, norm=norm, cmap='magma_r', origin='lower')
        ax.set_xticks(np.arange(1, size)/target_pix-0.5)
        ax.set_yticks(np.arange(1, size)/target_pix-0.5)
        ax.set_xlim(-0.5, psf_sci.shape[1]-0.5)
        ax.set_ylim(-0.5, psf_sci.shape[0]-0.5)
        
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid()
        ax.text(0.05, 0.05, '{0} {1}'.format(aor, label), fontsize=10,
                ha='left', va='bottom', transform=ax.transAxes, 
                bbox={'facecolor':'w', 'alpha':0.9, 'ec':'None'})
        
        fig.tight_layout(pad=0.1)
        if native_orientation:
            prod = '{0}-{1}-{2:.1f}.psfr'.format(aor, label, target_pix)
        else:
            prod = '{0}-{1}-{2:.1f}.psf'.format(aor, label, target_pix)
            
        fig.savefig(prod+'.png')
        pyfits.writeto(prod+'.fits'.format(aor, label), 
                       data=psf_sci, header=hdu.header, overwrite=True)

def init_model_attr(model):
    """
    Initialize pscale attribute of a model object
    """
    if not hasattr(model, 'pscale'):
        model.pscale = OrderedDict()
        for k in model.param_names:
            model.pscale[k] = 1.
    else:
        for k in model.param_names:
            if k not in model.pscale:
                model.pscale[k] = 1.
    
    return model

class ModelPSF(object):
    """
    PSF using `~astropy.modeling`
    """
    def __init__(self, model, input_scale=0.1, rmax=14, output_scale=0.1):
        self.input_scale = input_scale
        self.output_scale = output_scale
        self.rmax = rmax
        
        if 'amplitude' in model.param_names:
            model.fixed['amplitude'] = True
            
        init_model_attr(model)
        self.model = model
        self.initialize_output(rmax=rmax, output_scale=output_scale)
        
    def initialize_output(self, rmax=14, output_scale=0.1):
        if hasattr(rmax, '__len__'):
            xp = np.arange(-rmax[0], rmax[0],
                           output_scale/self.input_scale)
            yp = np.arange(-rmax[1], rmax[1],
                           output_scale/self.input_scale)
        else:
            xp = np.arange(-rmax, rmax+1.e-7, output_scale/self.input_scale)
            yp = xp
        
        self.rmax = rmax   
        self._xp, self._yp = np.meshgrid(xp, yp)
    
    def evaluate_psf(self, scale=0.1, rmax=14, ra=228.7540568, dec=-15.3806666, min_count=5, clip_negative=True, transform=None, warp_args={'order':3, 'mode':'constant', 'cval':0.}):
        """
        Dummy function to just evaluate the model
        """
        if (scale != self.output_scale) | (rmax != self.rmax):
            self.initialize_output(rmax=rmax, output_scale=scale)
        
        model = self.model(self._xp, self._yp)
        
        if transform is not None:
            model = warp_image(transform, model, warp_args=warp_args)
        
        model /= model.sum()
        return model, None, None
        
    @property
    def theta_index(self):
        
        idx = []
        for i, p in enumerate(self.model.fixed):
            if not self.model.fixed[p]:
                idx.append(i)
        
        return np.array(idx)
    
    @property
    def theta_labels(self):
        
        labels = []
        for i, p in enumerate(self.model.fixed):
            if not self.model.fixed[p]:
                labels.append(p)
        
        return labels
    
    @property
    def theta(self):
        return np.array([self.model.parameters[i] for i in self.theta_index])
    
    def set_theta(self, theta):
        for i, idx in enumerate(self.theta_index):
            self.model.parameters[idx] = theta[i]
            
    @property
    def theta_scale(self):
        return np.array([self.model.pscale[p] for p in self.theta_labels])
    
    def __repr__(self):
        return self.model.__repr__()
            
    @staticmethod    
    def init_model(type='Moffat2D'):
        from astropy.modeling.models import Moffat2D, Gaussian2D
        if type == 'Moffat2D':
            m = Moffat2D()
            m.fixed['amplitude'] = True
            init_model_attr(m)
            m.pscale['alpha'] = 0.1
        else:
            m = Gaussian2D()
            m.fixed['amplitude'] = True
            init_model_attr(m)
        
        return m
    
    def fit_cutout(self, cutout, minimize_kwargs={'method':'powell'}):
        from scipy.optimize import minimize
        pixel_shape = cutout.shape[::-1]
        
        rmax = [p/2 for p in pixel_shape]
        self.initialize_output(rmax=rmax, output_scale=self.input_scale)
        
        x0 = self.theta / self.theta_scale
        
        args = (self, cutout)
        _x = minimize(self._objective_function_psf, x0, args=args, **minimize_kwargs)
        #minimize(_objective_function_psf, x0, args=args, **minimize_kwargs)
        
        # model, _, _ = self.evaluate_psf(rmax=self.rmax)
        # a = (model*cutout).sum()/(model**2).sum()
        # resid = cutout - model*a
        # 
    @staticmethod
    def _objective_function_psf(theta_scl, psf, cutout):
        
        psf.set_theta(theta_scl*psf.theta_scale)
        
        model, _, _ = psf.evaluate_psf(rmax=psf.rmax)
        a = (model*cutout).sum()/(model**2).sum()
        resid = cutout - model*a
        
        chi2 = (resid**2).sum()
        print(theta_scl*psf.theta_scale, chi2)
        return chi2

MIPS_PSF_SCALE = 0.498
            
class FitsPSF(object):
    def __init__(self, psf_file='/Users/gbrammer/Research/grizli/CONF/mips_24_100K.fits', scale=0.1, rmax=14, native_scale=MIPS_PSF_SCALE):
        import grizli
        
        self.im = pyfits.open(psf_file)
        self.data = self.im[0].data / self.im[0].data.sum()
        
        self.native_scale = native_scale
        #self.native_scale = 0.498 #0.5 #0.498
        #self.native_scale = MIPS_PSF_SCALE #0.5 #0.498
        
        self.scale = -1
        self.rmax = -1
        
        # CoG
        yp, xp = np.indices(self.data.shape)
        
        xc = (self.data*xp).sum()
        yc = (self.data*yp).sum()
        self.xci = int(np.round(xc))
        self.yci = int(np.round(yc))
        
        self.R = np.sqrt((xp-xc)**2+(yp-yc)**2)*self.native_scale
        
        psf, _, _ = self.evaluate_psf(scale=scale, rmax=rmax)
    
    def curve_of_growth(self):
        so = np.argsort(self.R.flatten())
        cog = np.cumsum(self.data.flatten()[so])
        return self.R.flatten()[so], cog
        
    def evaluate_psf(self, scale=0.1, rmax=14, ra=228.7540568, dec=-15.3806666, min_count=5, clip_negative=True, transform=None, warp_args={'order':3, 'mode':'constant', 'cval':0.}):
        from photutils import resize_psf
        
        if (scale == self.scale) & (rmax == self.rmax):
            return self.psf, None, None
        
        self.scale = scale
        self.rmax = rmax
        
        N = int(np.ceil(rmax/self.native_scale*2)/2)
        self.psf0 = (self.data*(self.R <= rmax))[self.yci-N:self.yci+N, self.xci-N:self.xci+N]
        self.apcorr = 1./self.psf0.sum()
        
        psf = resize_psf(self.psf0, input_pixel_scale=self.native_scale, output_pixel_scale=scale)
        
        # bad interpolated values
        psfx = resize_psf((self.psf0 > 0)*1., input_pixel_scale=self.native_scale, output_pixel_scale=scale)
        msk = (psfx < 0.9*np.percentile(psfx, 90))
        psf = psf*(~msk)
        psf *= 1./psf.sum()
        self.psf = psf
        
        return psf, None, None


def MipsPSF(**kwargs):
    """
    FitsPSF using MIPS 100k tabulated PSF
    """
    psf_file = '/Users/gbrammer/Research/grizli/CONF/mips_24_100K.fits'
    if not os.path.exists(psf_file):
        psf_path = os.path.join(os.path.dirname(__file__), 'data/psf/')
        psf_file = os.path.join(psf_path, 'mips_24_100K.fits')
        
    return FitsPSF(psf_file=psf_file, scale=0.1, rmax=14, native_scale=MIPS_PSF_SCALE)


class MixturePSF(object):
    def __init__(self, N=3, rmin=1, rmax=10, rs=None, ee1=0., ee2=0., window=None):
        """
        Mixture of Gaussians PSF from `~tractor`.
        """
        from tractor.psf import GaussianMixtureEllipsePSF
        from tractor.ellipses import EllipseESoft

        self.N = N
        if rs is None:
            #rs = [np.log(2 * (r+1)) for r in range(N)]
            rs = np.linspace(np.log(rmin), np.log(rmax), N)

        self.mogs = [GaussianMixtureEllipsePSF(np.array([1]), np.array([[0., 0.]]), [EllipseESoft(r, ee1, ee2)]) for r in rs]
        self.coeffs = np.ones(N)
        self.indices = np.arange(1,6)
        self.i0 = self.indices*1
        self.p0 = self.getParams()*1
        self.coeffs = np.ones(self.N)
        self.orig_pixscale = np.nan

        self.window = window

        self.set_pixelgrid(size=32, instep=0.1, outstep=0.1, oversample=1)
        self.x0 = np.array([0., 0.])
        
        # Parameter bounds for fitting
        self.bounds_amp = (-1, 1)
        self.bounds_x = (-3, 3)
        self.bounds_y = (-3, 3)
        self.bounds_logr = (np.log(0.5*rmin), np.log(2*rmax))
        self.bounds_ee1 = (-1, 1)
        self.bounds_ee2 = (-1, 1)
        
    @property
    def bounds(self):
        """
        Parameter bounds
        """
        b = [self.bounds_amp,  # amplitude
             self.bounds_x,    # meanx0
             self.bounds_y,    # meany0,
             self.bounds_logr, # logr0
             self.bounds_ee1,  # ee1 ellip parameters
             self.bounds_ee2]  # ee2

        lo = [b[i][0] for i in self.indices]*self.N
        hi = [b[i][1] for i in self.indices]*self.N
        return (lo, hi)
    
    @staticmethod
    def ellipse_from_baphi(ba, phi):
        """
        Generate ellipse parameters e1, e2 from ba=b/a and phi PA in degrees
        """
        ab = 1. / ba
        e = (ab - 1) / (ab + 1)
        angle = np.radians(2. * (-phi))
        e1 = e * np.cos(angle)
        e2 = e * np.sin(angle)
        return e1, e2
        
    def setParams(self, theta):
        """
        Set parameters for each element in `self.mogs`.
        """
        Ni = self.indices.size
        thetas = np.reshape(theta, (-1, Ni))
        for i in range(self.N):
            p0 = np.array(self.mogs[i].getParams())*1
            p0[self.indices] = thetas[i]
            self.mogs[i].setParams(p0)

    def getParams(self):
        """
        Return a combined list of all `self.mogs` parameteres.
        """
        return np.hstack([np.array(m.getParams())[self.indices] for m in self.mogs])
    
    
    def get_matrix(self, pos, coeffs=None):
        """
        Evaluatee `self.mogs` at `pos` positions and return a matrix 
        suitable for least squares
        """
        _mog = np.stack([mog.mog.evaluate(pos) for mog in self.mogs]).T        
        _mog2 = (_mog/_mog.sum(axis=0))#.reshape((sh[0], sh[1], self.N))
        if coeffs is not None:
            return _mog2.dot(coeffs)            
        else:
            return _mog2


    def set_pixelgrid(self, size=32, instep=0.168, outstep=0.2, oversample=2):
        """
        Set the `self.pos` pixel grid for evaluating the output PSFs.  
        
        instep : pixel size where PSF was generated
        outstep : pixel size of final mosaic in the same input filter
        oversample : factor by which you want to oversample the output grid.  
        """
        ostep = outstep/instep/oversample
        xarr = np.arange(-size, size, 1) * ostep
        xp, yp = np.meshgrid(xarr, xarr)           
        self.shape = xp.shape
        self.oversample = oversample
        self.pos = np.array([xp.flatten(), yp.flatten()]).T 


    def evaluate_psf(self, transform=None, normalize=True, **kwargs):
        """
        Function to evaluate the PSF on the `self.pos` grid.  
                
        First two elements of `transform` used as a shift.
        
        """
        x0 = self.x0*1
        if transform is not None:
            x0 += transform[:2]/self.oversample

        psf = self.get_matrix(self.pos-x0, coeffs=self.coeffs)
        psf = psf.reshape(self.shape)

        if self.window is not None:
            psf *= self.window

        if normalize:
            psf /= psf.sum()

        return psf, 1., 1.


    @property 
    def centroid(self):
        """
        PSF centroid
        """
        psf, _, _ = self.evaluate_psf(transform=None)
        center = (self.pos.T*psf.flatten()).sum(axis=1)
        return center

    def recenter(self):
        """
        Shift gaussian centers to enforce centroid = (0,0)
        """
        self.x0 -= self.centroid
    
    def initialize_for_hscy(self, r=2.0, phi=0., ba=0.05, comp=-1):
        """
        Set last component to elongated ellipse for use with HSC-y band
        """
        e1, e2 = self.ellipse_from_baphi(ba, phi)
        pars = [0.5, 0., 0., r, e1, e2]
        self.mogs[comp].setAllParams(pars) 
        
    def from_cutout(self, input, mask=None, pixscale=None, show=True, fit_ellipse=True, verbose=True, center_first=True, lsq_kwargs={'ftol':1.e-8, 'xtol':1.e-8, 'gtol':1.e-8, 'loss':'arctan', 'method':'trf', 'max_nfev':200}, indices=None, **kwargs):
        """
        Generate a MixturePSF from a cutout.
        
        input : string FITS filename or array
        show : make a figure showing the fit
        
        """
        from scipy.optimize import least_squares, minimize 
        import matplotlib.pyplot as plt
        import astropy.wcs as pywcs
        
        if isinstance(input, str):
            psf = pyfits.open(input)
            cutout = psf[0].data
            
            if mask is -1:
                mask = cutout > 0
                
            try:
                wcs = pywcs.WCS(psf[0].header)
                self.orig_pixscale = utils.get_wcs_pscale(wcs)
            except:
                if pixscale is not None:
                    self.orig_pixscale = pixscale
                    
        else:
            cutout = input*1.
            if pixscale is not None:
                self.orig_pixscale = pixscale
                
        xarr = np.arange(0, cutout.shape[0], 1.) 
        xarr -= cutout.shape[0]/2.
        xpo, ypo = np.meshgrid(xarr, xarr)           
        pos = np.array([xpo.flatten(), ypo.flatten()]).T 

        if center_first:
            # Centroid
            cx = (xpo*cutout).sum()/cutout.sum()
            cy = (ypo*cutout).sum()/cutout.sum()
            for mog in self.mogs:
                mog.meanx0 = cx#+1
                mog.meany0 = cy#+1

            # Centers
            if verbose > 0:
                print('\n\nFit centers\n\n')

            self.indices = np.arange(1,3)
            args = (self, cutout, mask, pos, np.linalg.lstsq, verbose, 'lm')
            margs = (self, cutout, mask, pos, np.linalg.lstsq, verbose, 'model')
            xi = self.getParams()
            _x0 = least_squares(self._objmog, xi, bounds=self.bounds, args=args, **lsq_kwargs)
            x0 = _x0.x
            a0, m0 = self._objmog(x0, *margs)
            self.coeffs = a0[1:]
            
        # Fit all params
        if verbose > 0:
            print('\n\nFit full\n\n')
            
        if indices is None:
            self.indices = np.arange(1+2*center_first, 4+fit_ellipse*2)
        else:
            self.indices = indices
            
        args = (self, cutout, mask, pos, np.linalg.lstsq, verbose, 'lm')
        margs = (self, cutout, mask, pos, np.linalg.lstsq, verbose, 'model')
        xi = self.getParams()
        _x3 = least_squares(self._objmog, xi, bounds=self.bounds, args=args, **lsq_kwargs)
        x3 = _x3.x
        a3, m3 = self._objmog(x3, *margs)
        self.coeffs = a3[1:]

        ### Show
        if show:
            bg = 0.

            fig = plt.figure(figsize=(9,3))
            ax = fig.add_subplot(131)
            imsh = ax.imshow(np.log10(cutout-bg))
            vmin, vmax= imsh.get_clim()

            mog_model = m3

            ax = fig.add_subplot(132)
            ax.imshow(np.log10(mog_model), vmin=vmin, vmax=vmax)

            ax = fig.add_subplot(133)
            ax.imshow(np.log10(cutout - bg - mog_model), vmin=vmin, vmax=vmax)

            for ax in fig.axes:
                ax.set_xticklabels([])
                ax.set_yticklabels([])

            fig.tight_layout(pad=0.1)


    def to_header(self):
        """
        Parameters as FITS header
        """
        h = pyfits.Header()

        pars = self.getParams()
        
        h['PSFN'] = (self.N, 'MixturePSF N components')
        for i in range(self.N):
            h[f'PSFC{i}'] = self.coeffs[i]
            pars = self.mogs[i].getParams()
            for j in range(6):
                h[f'PSFP{i}_{j}'] = pars[j]

        h['OPSCALE'] = self.orig_pixscale, 'Original pixel scale'
        return h


    @staticmethod
    def from_header(header, initial_size=32, hires_pscale=0.1):
        """
        Initialize from parameters in a `~astropy.io.fits.Header`
        """
        psf_obj = MixturePSF(N=header['PSFN']) 
        
        for i in range(psf_obj.N):
            psf_obj.coeffs[i] = header[f'PSFC{i}']
            pars = [header[f'PSFP{i}_{j}'] for j in range(6)]
            psf_obj.mogs[i].setParams(pars)
        
        orig_pix = header['OPSCALE']
        if hires_pscale is None:
            hires_pscale = orig_pix
            
        psf_obj.set_pixelgrid(size=initial_size, instep=orig_pix,
                              outstep=orig_pix, 
                              oversample=orig_pix/hires_pscale)
        return psf_obj


    @staticmethod        
    def _objmog(theta, mixpsf, cutout, mask, pos, lsq, verbose, ret):
        mixpsf.setParams(theta)
        cflat = cutout.flatten()
        _X = np.hstack([cflat[:,None]*0.+1, mixpsf.get_matrix(pos)])

        _res = lsq(_X, cutout.flatten())
        model = _X.dot(_res[0])

        if ret == 'model':
            return _res[0], model.reshape(cutout.shape) - _res[0][0]

        chi2 = (model - cflat)
        if mask is not None:
            chi2 = chi2[mask.flatten()]
            
        if (verbose & 2) > 0:
            print(theta, (chi2**2).sum())

        if ret.startswith('chi2'):
            return (chi2**2).sum()
        elif ret.startswith('lm'):
            return chi2
        else:
            return -0.5*(chi2**2).sum()
        
class ArrayPSF(object):
    def __init__(self, psf_data=None, resample=1):
        
        self.psf_data = psf_data/psf_data.sum()
        if resample > 0:
            sh = self.psf_data.shape
            new_psf = np.zeros((sh[0]*resample, sh[1]*resample), dtype=self.psf_data.dtype)
            for i in range(resample):
                for j in range(resample):
                    new_psf[i::resample, j::resample] += self.psf_data
            
            self.psf_data = new_psf/resample**2
            
    def evaluate_psf(self, ra=228.7540568, dec=-15.3806666, min_count=5, clip_negative=True, transform=None, warp_args={'order':3, 'mode':'constant', 'cval':0.}):
        
        psf = self.psf_data
        if transform is not None:
            psf = warp_image(transform, psf, warp_args=warp_args)
            
        return psf/psf.sum(), 1., 1.
    
    def get_exposure_time(self, ra, dec, get_files=False, verbose=False):
        return 1., 1.
        
class IracPSF(object):
    """
    Tools for generating an IRAC PSF at an arbitrary location in a mosaic
    """
    def __init__(self, ch=1, scale=0.1, verbose=True, aor_list=None, avg_psf=None, use_zodi_weight=True):
        self.psf_data = OrderedDict()
        self.psf_arrays = OrderedDict()
        self.ch = ch
        self.scale = scale
        self.avg_psf = avg_psf

        self.warm_weight = {1:7, 2:1}
        
        zodi_file = get_zodi_file()
        if os.path.exists(zodi_file) & use_zodi_weight:
            self.zodi_hdu = pyfits.open(zodi_file)
        else:
            self.zodi_hdu = None
            
        _ = self.load_psf_data(ch=ch, scale=scale, verbose=verbose, aor_list=aor_list, avg_psf=self.avg_psf)
        self.psf_data, self.psf_arrays = _
        
            
    @property
    def N(self):
        return len(self.psf_data)
    
    @property 
    def aor_names(self):
        return list(self.psf_data.keys())
        
    def load_psf_data(self, ch=1, scale=0.1, verbose=True, aor_list=None, avg_psf=None):
    
        import glob
        
        from grizli import utils
        #from matplotlib.patches import Polygon
        from matplotlib.path import Path
        from scipy.spatial import ConvexHull
        
        files = glob.glob('*ch{0}.log.fits'.format(ch))
        files.sort()
    
        psf_data = OrderedDict()
        for file in files:
            aor = file.split('-ch')[0]
            if aor_list is not None:
                if aor not in aor_list:
                    continue

            log = utils.read_catalog(file)
            
            if self.zodi_hdu is not None:
                has_zodi = True
                coo = np.mean(log['crval'], axis=0)
                zlev, zerr = get_spitzer_zodimodel(ra=coo[0], dec=coo[1], 
                                                   zodi_file=self.zodi_hdu, 
                                                   mjd=log['mjd_obs'],
                                                   ch=ch, 
                                                   exptime=log['exptime'])
                log['weight'] = 1/zerr**2
            else:
                log['weight'] = log['exptime']
                
                has_zodi = False
                
            psf_files = [f'{aor}-ch{ch}-{scale:.1f}.{ext}.fits' 
                         for ext in ['psf','psfr']]
                                                         
            psf_file = '(N/A)'
            for pfile in psf_files:
                if os.path.exists(pfile):
                    psf_file = pfile
                    break
                    
            if verbose:
                print(f'load_psf_data: log={file} zodi={has_zodi} psf={psf_file}')
            
            if os.path.exists(psf_file):    
                psf = pyfits.open(psf_file)
                psf_mask = (psf[0].data != 0)
                psf_image = psf[0].data
                
            else:
                psf_image = None
                psf_mask = None
                
            footprints = []
            for i, coo in enumerate(log['corners']):
                footprints.append(Path(coo))
            
            full_path = Path.make_compound_path(*footprints)
            hull = ConvexHull(full_path.vertices)
            full_footprint = Path(full_path.vertices[hull.vertices,:])
            
            psf_data[aor] = {'log':log,  'psf_file':psf_file, 
                             'psf_image':psf_image, 'psf_mask':psf_mask, 
                             'bcd_footprints':footprints, 
                             'footprint': full_footprint}
        
        # Do we need to rotate individual PSFs?
        for k in psf_data:
            if 'psfr.fits' in psf_data[k]['psf_file']:        
                theta = np.mean(psf_data[k]['log']['theta'])
                psf_i = psf_data[k]['psf_image']
                msk_i = psf_data[k]['psf_mask']
                
                try:
                    warped = warp_image(np.array([0,0,-theta,1]), psf_i)
                    mwarped = warp_image(np.array([0,0,-theta,1]), msk_i)
                except ValueError:
                    psf_i = psf_i.byteswap().newbyteorder()
                    warped = warp_image(np.array([0,0,-theta,1]), psf_i)
                    mwarped = warp_image(np.array([0,0,-theta,1]), msk_i)
                
                coswindow = CosineBellWindow(alpha=1)
                window = coswindow(warped.shape)**0.5 #0.05
                warped *= window
                warped /= warped.sum()
                
                psf_data[k]['psf_image'] = warped
                psf_data[k]['psf_mask'] = np.isfinite(warped)
                psf_data[k]['psf_mask'] &= warped/warped.max() > 1.e-5
                psf_data[k]['psf_image'][~psf_data[k]['psf_mask']] = 0.
                
        if avg_psf is None:
            img = np.array([psf_data[k]['psf_image'].flatten() 
                            for k in psf_data])
            msk = np.array([psf_data[k]['psf_mask'].flatten() 
                            for k in psf_data])
            psf_shape = psf[0].data.shape
        else:
            print('Use rotated `avg_psf`.')
            img = []
            msk = []
            for k in psf_data:
                theta = np.mean(psf_data[k]['log']['theta'])
                try:
                    warped = warp_image(np.array([0,0,-theta,1]), avg_psf)
                except ValueError:
                    avg_psf = avg_psf.byteswap().newbyteorder()
                    warped = warp_image(np.array([0,0,-theta,1]), avg_psf)
                    
                img.append(warped.flatten()/warped.sum())
                msk.append(img[-1] != 0)
                
            img = np.array(img)
            msk = np.array(msk)
            psf_shape = avg_psf.shape
        
        masked = img*msk
        psf_arrays = {'image':img, 'mask':msk, 'shape':psf_shape, 
                      'masked':masked}
        return psf_data, psf_arrays


    def drizzle_psf(self, ra=228.7540568, dec=-15.3806666, min_count=5, clip_negative=True, transform=None, warp_args={'order':3, 'mode':'constant', 'cval':0.}):
        """
        Drizzle a model PSF rather than rotating & resampling [TBD]
        """
        pass


    def evaluate_psf(self, ra=228.7540568, dec=-15.3806666, min_count=5, clip_negative=True, transform=None, warp_args={'order':3, 'mode':'constant', 'cval':0.}):
        """
        Evaluate weighted PSF at a given location using saved PSFs for each AOR
        
        transform: `~skimage.transform.SimilarityTransform`
        
        """
        import numpy as np
        from skimage.transform import warp
        
        exptime = np.zeros(self.N)
        count = np.zeros(self.N, dtype=int)
        expwht = np.zeros(self.N)
        
        point = (ra, dec)
        for i, aor in enumerate(self.psf_data):
            if not self.psf_data[aor]['footprint'].contains_point(point):
                #print('skip', aor); break
                continue
                
            for j, pat in enumerate(self.psf_data[aor]['bcd_footprints']):
                if pat.contains_point(point):
                    #print(i, j)
                    exptime[i] += self.psf_data[aor]['log']['exptime'][j]
                    expwht[i] += self.psf_data[aor]['log']['weight'][j]
                    count[i] += 1
        
        count_mask = (count > min_count)      
        if count_mask.sum() == 0:
            return np.zeros(self.psf_arrays['shape']), exptime, count
        
        # Background higher in warm mission.  Roughly determined from GOODSN
        mjd0 = np.array([self.psf_data[aor]['log']['mjd_obs'][0] 
                        for aor in self.psf_data])
        
        # if self.ch in self.warm_weight:
        #     warm_scale = 1+(mjd0 > 54963.0)*self.warm_weight[self.ch]
        # else:
        #     warm_scale = 1.
        warm_scale = 1.
            
        # if self.ch == 1:
        #     warm_scale = 1+(mjd0 > 54963.0)*7
        # elif self.ch == 2:
        #     warm_scale = 1+(mjd0 > 54963.0)*1
        # else:
        #     warm_scale = 1.
            
        #expwht = (exptime/warm_scale)[count_mask]
        #num = (self.psf_arrays['image']*self.psf_arrays['mask'])
        num = self.psf_arrays['masked'][count_mask].T.dot(expwht[count_mask])
        den = (self.psf_arrays['mask'][count_mask]).T.dot(expwht[count_mask])
        
        psf = (num/den).reshape(self.psf_arrays['shape'])
        psf[~np.isfinite(psf)] = 0
        
        if clip_negative:
            psf[psf < 0] = 0
        
        if transform is not None:
            psf = warp_image(transform, psf, warp_args=warp_args)
            
        return psf/psf.sum(), exptime, count


    def get_exposure_time(self, ra, dec, get_files=False, verbose=False):
        """
        Compute number of exposures and total exposure time for an array of 
        positions
        """
        N = len(np.atleast_1d(ra))
        points = np.array([np.atleast_1d(ra), np.atleast_1d(dec)]).T
        count = np.zeros(N, dtype=np.int32)
        exptime = np.zeros(N, dtype=np.float32)
        
        idx = np.arange(N)
        files = []
        for i, aor in enumerate(self.psf_data):
            if verbose:
                print('get_exposure_time {0} ({1}/{2})'.format(aor, i, self.N))
            
            clip = self.psf_data[aor]['footprint'].contains_points(points)
            if clip.sum() == 0:
                continue
                
            for j, pat in enumerate(self.psf_data[aor]['bcd_footprints']):
                test = pat.contains_points(points[clip])
                count[idx[clip][test]] += 1
                exptime[idx[clip][test]] += self.psf_data[aor]['log']['exptime'][j]
                
                if get_files:
                    if test.sum() > 0:
                        files.append(self.psf_data[aor]['log']['file'][j])
                        
        if get_files:
            return count, exptime, files
        else:
            return count, exptime
        
    def weight_map(self, hdu, sample=4, make_hdulist=True, verbose=True):
        """
        Create exposure time and count maps based on WCS from an input HDU/wcs
        """
        sh = hdu.data.shape
        newsh = (sh[0]//sample, sh[1]//sample)
        
        if hasattr(hdu, 'header'):
            wcs = pywcs.WCS(hdu.header, relax=True)
        else:
            wcs = hdu
        
        yp, xp = np.indices(newsh)*sample 
        rp, dp = wcs.all_pix2world(xp, yp, 0) 
        nexp, expt = self.get_exposure_time(rp.flatten(), dp.flatten(), verbose=verbose)
        
        nexp = nexp.reshape(newsh)
        expt = expt.reshape(newsh)
        
        new_header = utils.to_header(wcs)
        for k in new_header:
            if k.startswith('CDELT'):
                continue
            
            if k.startswith('CD'):
                new_header[k] *= sample
            elif k.startswith('CRPIX'):
                new_header[k] = new_header[k]/sample 
                
        new_header['NAXIS1'] = newsh[1]
        new_header['NAXIS2'] = newsh[0]
        
        # Match lower left pixel
        new_wcs = pywcs.WCS(new_header)
        p = [-0.5, -0.5]
        xy = new_wcs.all_world2pix(wcs.all_pix2world(np.array([p]), 0), 0)
        new_header['CRPIX1'] -= xy.flatten()[0]+0.5
        new_header['CRPIX2'] -= xy.flatten()[1]+0.5
        
        new_header['RESAMP'] = (sample, 'Sample factor from parent image')
        
        if make_hdulist:
            hdul = pyfits.HDUList()
            hdul.append(pyfits.ImageHDU(data=expt, header=new_header, name='EXPTIME'))
            hdul.append(pyfits.ImageHDU(data=nexp, header=new_header, name='NEXP'))
            return hdul
        else:
            return new_header, expt, nexp


def warp_catalog(transform, xy, image, center=None):
    
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
        
    
    tf_rscale = SimilarityTransform(translation=trans/scale, rotation=None, scale=scale)
    shifted = tf_rscale(xy)
    
    tf_rot = SimilarityTransform(translation=[0, 0], rotation=-rot/180*np.pi, 
                                 scale=1)
    rotated = tf_rot(shifted-center)+center
    
    return rotated


def combine_products(wcsfile='r15560704/ch1/bcd/SPITZER_I1_15560704_0036_0000_6_wcs.fits', skip_existing=True):
    """
    Make a single multi-extension FITS file
    """
    import astropy.io.fits as pyfits
    import numpy as np
    
    if not wcsfile.endswith('_wcs.fits'):
        print('Need _wcs.fits')
        return False
    
    outfile = wcsfile.replace('_wcs.fits', '_xbcd.fits.gz')
    if os.path.exists(outfile) & skip_existing:
        print('Skip {0}'.format(wcsfile))
        
    wcs = pyfits.open(wcsfile)
    if 'SPITZER_M' in wcsfile:
        bcd_file = glob.glob(wcsfile.replace('_wcs.fits','*bcd.fits'))[0]
        bunc_file = glob.glob(wcsfile.replace('_wcs.fits','*bunc.fits'))[0]
        cbcd = pyfits.open(bcd_file)
        cbunc = pyfits.open(bunc_file)
    else:
        cbcd = pyfits.open(wcsfile.replace('_wcs.fits','_cbcd.fits'))
        cbunc = pyfits.open(wcsfile.replace('_wcs.fits','_cbunc.fits'))
            
    hdul = pyfits.HDUList([cbcd[0], pyfits.ImageHDU(data=cbunc[0].data, header=cbunc[0].header), pyfits.ImageHDU(data=wcs[0].data, header=wcs[0].header)])
    hdul[0].header['EXTNAME'] = 'CBCD'
    hdul[1].header['EXTNAME'] = 'CBUNC'
    hdul[2].header['EXTNAME'] = 'WCS'
    
    dq = np.cast[np.int16](wcs[0].data*1)
    for ext in ['bimsk','brmsk','ebmsk']:
        mask_file = wcsfile.replace('_wcs.fits','_{0}.fits'.format(ext))
        if os.path.exists(mask_file):
            bmsk = pyfits.open(mask_file)
            dq |= bmsk[0].data
    
    hdul[2].data = dq
    print(outfile)
    hdul.writeto(outfile, overwrite=True, output_verify='fix')
            
def evaluate_irac_psf(ch=1, scale=0.1, ra=228.7540568, dec=-15.3806666, psf_data=None):
    """
    Evaluate weighted PSF at a given location using saved PSFs for each AOR
    """
    import glob
    from grizli import utils
    from matplotlib.patches import Polygon
    
    files = glob.glob('*ch{0}.log.fits'.format(ch))
    files.sort()
    
    psf_num = None
    psf_wht = 0
    exp_count = 0
    for file in files:
        tab = utils.read_catalog(file)
        aor = file.split('.ch')[0]
        psf = pyfits.open('{0}-ch{1}-{2:.1f}.psf.fits'.format(aor, ch, scale))
        psf_mask = (psf[0].data != 0)
        for i, coo in enumerate(tab['corners']):
            pat = Polygon(coo)
            if pat.contains_point([ra, dec]):
                wht_i = tab['exptime'][i]*psf_mask
                if psf_num is None:
                    psf_num = psf[0].data*wht_i
                else:
                    psf_num += psf[0].data*wht_i
                
                psf_wht += wht_i                   
                exp_count += 1
                
    out_psf = psf_num/psf_wht
    out_psf[psf_wht == 0] = 0
    return out_psf, exp_count
                
        
        