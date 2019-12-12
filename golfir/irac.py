"""
Processing IRAC BCDs
"""

import glob
import os

import numpy as np
import numpy.ma

import astropy.io.fits as pyfits
import astropy.wcs as pywcs
import astropy.units as u

from grizli import utils, prep

def process_all(channel='ch1', output_root='irac', driz_scale=0.6, kernel='point', pixfrac=0.5, out_hdu=None, wcslist=None, pad=10, radec=None, aor_ids=None, use_brmsk=True, nmin=5, flat_background=False, two_pass=False, min_frametime=20, instrument='irac', run_alignment=True, align_threshold=3, mips_ext='_bcd.fits'):
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
        
    aors = {}
    pop_list = []
    N = len(aor_ids)
    for i, aor in enumerate(aor_ids):
        files = glob.glob('{0}/{1}/bcd/{2}'.format(aor, channel, inst_key))
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
                aors[aor].align_to_reference(reference=['GAIA'], radec=radec, 
                                             threshold=align_threshold)
            except:
                fp = open('{0}-{1}_wcs.failed'.format(aor, aors[aor].label),'w')
                fp.write(time.ctime())
                fp.close()
                
                pop_list.append(aor)
        
    if len(aors) == 0:
        return {}
            
    for aor in pop_list:
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
                
            print(i, aor, aors[aor].N)
            aors[aor].drz = aors[aor].drizzle_simple(wcslist=wcslist, driz_scale=driz_scale, theta=0, kernel=kernel, out_hdu=out_hdu, pad=pad, pixfrac=pixfrac, flat_background=flat_background, force_pedestal=force_pedestal, extra_mask=extra_mask)#'point')
        
        if 0:
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
            bkg_params = {'bh': 16, 'bw': 16, 'fh': 3, 'fw': 3, 'pixel_scale': 0.5+1.5*(instrument == 'mips')}
            
            apertures = np.logspace(np.log10(0.3), np.log10(12), 5)*u.arcsec
            detect_params = prep.SEP_DETECT_PARAMS.copy()

            cat = prep.make_SEP_catalog(root=irac_root, threshold=1.5, 
                            get_background=True, bkg_params=bkg_params, 
                            phot_apertures=apertures, aper_segmask=True,
                            column_case=str.lower, 
                            detection_params=detect_params, 
                            pixel_scale=driz_scale)
            
            seg = pyfits.open('{0}_seg.fits'.format(irac_root))[0]
            seg_wcs = pywcs.WCS(seg.header)
            seg_wcs.pscale = utils.get_wcs_pscale(seg_wcs)
            for i, aor in enumerate(aors):
                print('\n {0} {1} \n'.format(i, aor))
                blot_seg = aors[aor].blot(seg_wcs, seg.data, interp='poly5')
                med = aors[aor].med2d[0]-aors[aor].pedestal[0]
                aors[aor].extra_mask = (blot_seg > 0) #| (med < -0.05)
                
    for i, aor in enumerate(aors):
        aors[aor].save_state(mips_ext=mips_ext, verbose=True)
        aors[aor].tab = aors[aor].wcs_table()
        
    return aors
    
    # PSF
    psf_coords = (340.8852003, -9.604230278)
    for i, aor in enumerate(aors):
        print('PSF', i, aor, aors[aor].N)
        aors[aor].drz = aors[aor].drizzle_simple(wcslist=wcslist, driz_scale=driz_scale, theta=0, kernel=kernel, out_hdu=out_hdu, pad=pad, psf_coords=psf_coords, pixfrac=pixfrac)
        
        sci, wht, ctx, head, w = aors[aor].drz

        root_i = '{0}-{1}-{2}'.format(output_root, aor, output_label)
        pyfits.writeto('{0}_psf_sci.fits'.format(root_i), data=sci*100, header=head, overwrite=True)
        pyfits.writeto('{0}_psf_wht.fits'.format(root_i), data=wht, header=head, overwrite=True)

        
    ds9.frame(i0+1)
    ds9.view(drz_sci, header=head)
    
    # apers = [a*u.arcsec for a in [1,2,3,4,5,6,8,10,12,15,20]]
    # cat = prep.make_SEP_catalog(root=irac_root, threshold=1.4, get_background=True, bkg_only=False, bkg_params={'fw': 3, 'bw': 32, 'fh': 3, 'bh': 32}, verbose=True, sci=None, wht=None, phot_apertures=apers, rescale_weight=True, column_case=str.upper, save_to_fits=True, source_xy=None, autoparams=[2.5, 3.5], mask_kron=False, max_total_corr=2, err_scale=-np.inf)
    # 
    # ok = cat['FLAG_APER_7'] == 0
    # flux_ratio = cat['FLUX_APER_2']/cat['FLUX_APER_7']
    # plt.scatter(cat['MAG_AUTO'][ok], flux_ratio[ok], alpha=0.2)
    # 
    # Force photometry
    
    root='j112000p0641'
    root = os.getcwd().split('/')[-3]
    phot = utils.read_catalog('../{0}_phot.fits'.format(root))
    apers = [3*u.arcsec]
    apcorr = 1./0.70
    
    cat = prep.make_SEP_catalog(root=irac_root, threshold=1.4, get_background=True, bkg_only=False, bkg_params={'fw': 3, 'bw': 32, 'fh': 3, 'bh': 32}, verbose=True, sci=None, wht=None, phot_apertures=apers, rescale_weight=True, column_case=str.upper, save_to_fits=True, source_xy=(phot['ra'], phot['dec']), autoparams=[2.5, 3.5], mask_kron=False, max_total_corr=2, err_scale=-np.inf)
    
    phot['irac_ch1_flux'] = cat['FLUX_APER_0']*apcorr
    phot['irac_ch1_err'] = cat['FLUXERR_APER_0']*apcorr
    mask = cat['MASK_APER_0'] > 3
    phot['irac_ch1_flux'][mask] = -99
    phot['irac_ch1_err'][mask] = -99
    
    phot.meta['IRAC_APCORR'] = apcorr
    
    phot.write('../{0}_phot.fits'.format(root), overwrite=True)
    
    # Apcorr
    import astropy.io.fits as pyfits
    import numpy as np
    import stsci.convolve
    
    himg = pyfits.open('../j004400m2034-f160w_drz_sci.fits')
    psf = pyfits.open('/Users/gbrammer/Downloads/apex_core_irac_warm_PRFs/apex_sh_IRACPC1_col129_row129_x100.fits')
    N = 64
    psfd = psf[0].data[::5,::5][255-N:255+N,255-N:255+N] # ~0.06"
    psfd /= psfd.sum()
        
    #hsm = nd.convolve(himg[0].data, psfd)
    slx, sly = slice(x0,x0+2048), slice(y0,y0+2048)
    #data = himg[0].data[sly, slx]
    data = himg[0].data
    
    hsm = stsci.convolve.convolve2d(data, psfd, output=None, mode='nearest', cval=0.0, fft=1)
    
    import sep
    
    aper3 = 3/0.06
    flux, fluxerr, flag = sep.sum_circle(data.byteswap().newbyteorder(), 
                                  phot['x_image']-1, phot['y_image']-1,
                                  aper3/2, err=data*0.+1, 
                                  gain=2000., subpix=5)
    #
    fluxsm, fluxerr, flag = sep.sum_circle(hsm.astype('>f4').byteswap().newbyteorder(), 
                                  phot['x_image']-1, phot['y_image']-1,
                                  aper3/2, err=data*0.+1, 
                                  gain=2000., subpix=5)
#

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

    if False:
        ra, dec = 340.8852003, -9.604230278
        i=20
        wcs = self.wcs[i]
    
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
                
    def read_files(self, use_brmsk=True, min_frametime=20):
        
        self.hdu = [pyfits.open(file)[0] for file in self.files]
        for i in  np.arange(len(self.hdu))[::-1]:
            if self.hdu[i].header['EXPTIME'] < min_frametime:
                self.hdu.pop(i)
                self.files.pop(i)
        
        self.nskip = 0
        self.N = len(self.files)
        if self.N == 0:
            return None
            
        self.cbcd = np.stack([im.data*1 for im in self.hdu])

        self.cbunc = np.stack([pyfits.open(file.replace('bcd.', 'bunc.'))[0].data for file in self.files])
        if self.instrument == 'irac':
            self.bimsk = np.stack([pyfits.open(file.replace('_cbcd','_bcd').replace('bcd.', 'bimsk.'))[0].data for file in self.files])
        else:
            self.bimsk = np.stack([pyfits.open(file.replace('_cbcd','_bcd').replace('bcd.', 'bbmsk.').replace('_ebb','_eb'))[0].data for file in self.files])
            
        if use_brmsk:
            try:
                self.brmsk = np.zeros(self.bimsk.shape, dtype=np.int16)
                for i, file in enumerate(self.files):
                    rmsk_file =  file.replace('bcd.', 'brmsk.')
                    if os.path.exists(rmsk_file):
                        self.brmsk[i,:,:] = pyfits.open(rmsk_file)[0].data     
            except:
                self.brmsk = None
        else:
            self.brmsk = None
            
        self.wcs = [pywcs.WCS(im.header) for im in self.hdu]
        for w in self.wcs:
            w.pscale = utils.get_wcs_pscale(w)
        
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
    
    def make_median_image(self, pixel_scale=None, extra_mask=None):
        
        if pixel_scale is None:
            pixel_scale = self.wcs[0].pscale
        
        pedestal, med2d = self.get_median(extra_mask=extra_mask)
        
        med_hdu = utils.make_maximal_wcs(self.wcs, pixel_scale=pixel_scale, theta=-self.theta, pad=10, get_hdu=True, verbose=False)
        sh = med_hdu.data.shape
        med_wcs = pywcs.WCS(med_hdu.header)
        med_wcs.pscale = pixel_scale
        
        med_data = np.zeros((self.N, sh[0], sh[1]))
        for i in range(self.N):
            drz = utils.drizzle_array_groups([(self.cbcd-med2d)[i,:,:]], [(self.ivar*(self.dq == 0))[i,:,:]], [self.shift_wcs[i]], outputwcs=med_wcs, kernel='point')
            med_data[i,:,:] = drz[0]
        
        med_ma = np.ma.masked_array(med_data, mask=med_data == 0)
         
        med = np.ma.median(med_ma[self.nskip:,:,:], axis=0)
        
        return med_wcs, med.data
        
    def blot(self, med_wcs, med, interp='poly5'):
        from drizzlepac.astrodrizzle import ablot
        
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
    
    def pipeline(self, ds9=None):
        pass
        
        phot = utils.read_catalog('../j232508m1212_phot_apcorr.fits')
        hmag = 23.9-2.5*np.log10(phot['flux_auto_fix'])
        ref = phot[hmag < 22]
        
        med_wcs, med = self.make_median_image()
        
        merr = np.ones_like(med, dtype=np.float)*utils.nmad(med[med != 0])
        mcat, mseg = prep.make_SEP_catalog_from_arrays(med.byteswap().newbyteorder(), merr.byteswap().newbyteorder(), (med == 0), segmentation_map=True, threshold=1.1, wcs=med_wcs, get_background=True)
        
        so = np.argsort(mcat['flux'])
        ref = mcat#[so[:150]]
        
        blot_data = self.blot(med_wcs, med)        
        pedestal, med2d = self.get_median()
        nmad = self.get_nmad(median=med2d)
                
        # Add extra term in bright sources
        fmax = 0.5
        eblot = 1-np.clip(blot_data, 0, fmax)/fmax
        
        self.dq = self.orig_dq |  ((self.cbcd - med2d - blot_data)*np.sqrt(self.ivar)*eblot > 7)
        
        ds = self.get_shifts(ref=ref)
        self.shift_wcs = [utils.transform_wcs(self.wcs[i], translation=-self.shifts[i,:], rotation=-self.rot[i], scale=1.0) for i in range(self.N)]
        
        # Second iteration
        med_wcs2, med2 = self.make_median_image()
        blot_data2 = self.blot(med_wcs2, med2)
        
        ds9.frame(1)
        ds9.view(((self.cbcd - blot_data - med2d)*(self.dq == 0))[j,:,:])
        
        ds9.frame(2)
        ds9.view(((self.cbcd - blot_data2 - med2d)*(self.dq == 0))[j,:,:])
        
        for iter in range(3):
            fmax = 0.5
            eblot = 1-np.clip(blot_data2, 0, fmax)/fmax

            self.dq = self.orig_dq |  ((self.cbcd - med2d - blot_data)*np.sqrt(self.ivar)*eblot > 4.5)
        
            ds = self.get_shifts(ref=ref)
            self.shift_wcs = [utils.transform_wcs(self.wcs[i], translation=-self.shifts[i,:], rotation=-self.rot[i], scale=1.0) for i in range(self.N)]

            med_wcs2, med2 = self.make_median_image()
            blot_data2 = self.blot(med_wcs2, med2)
            
            ds9.frame(iter+3)
            ds9.view(((self.cbcd - blot_data2 - med2d)*(self.dq == 0))[j,:,:])
    
    def drizzle_simple(self, force_pedestal=True, dq_sn=5, fmax=0.5, flat_background=False, extra_mask=None, **kwargs):
        
        if (not hasattr(self, 'pedestal')) | force_pedestal: 
            med_wcs, med = self.make_median_image()        
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
        return np.sum([self.hdu[i].header['EXPTIME'] for i in range(self.nskip, self.N)])
    
    def get_all_psfs(self, psf_coords=None, func=get_bcd_psf):
        
        ra, dec = psf_coords
        full_psf = np.zeros((self.N, 256, 256))
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
                           kernel=kernel, pixfrac=pixfrac)
        
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
    
    def align_to_reference(self, reference=['GAIA','PS1'], radec=None, kernel='square', pixfrac=0.8, threshold=3, assume_close=True):
        from grizli import prep
        import astropy.units as u
        from drizzlepac import updatehdr
        
        res = self.drizzle_simple(dq_sn=5, fmax=0.5, driz_scale=1.0, kernel=kernel, pixfrac=pixfrac, theta=self.theta)
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
        radec_ext, ref_name = prep.get_radec_catalog(ra=ra, dec=dec, radius=10.0, product='{0}-{1}'.format(self.name, self.label), verbose=True, reference_catalogs=reference, use_self_catalog=False, date=self.hdu[0].header['MJD_OBS'], date_format='mjd')
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
            prep.table_to_radec(rd[idx][keep], 'tmp.radec')
            radec = 'tmp.radec'
            
        if self.instrument == 'mips':
            mag_limits = [10, 30]
        else:
            mag_limits = [14,25]
            
        ali = prep.align_drizzled_image(root='{0}-{1}'.format(self.name, self.label), mag_limits=mag_limits, radec=radec, NITER=3, clip=20, log=True, outlier_threshold=5, verbose=True, guess=[0.0, 0.0, 0.0, 1], simple=True, rms_limit=2, use_guess=False, triangle_size_limit=[5, 1800], triangle_ba_max=0.9, max_err_percentile=99)
        
        orig_wcs, drz_wcs, out_shift, out_rot, out_scale = ali
        
        self.alignment = out_shift, out_rot, out_scale
        
        for i, h in enumerate(self.hdu):
            print(i, self.files[i])
            tmpfile = 'tmp_{0}_cbcd.fits'.format(self.label)
            h.writeto(tmpfile, overwrite=True)
            updatehdr.updatewcs_with_shift(tmpfile, 
                                '{0}-{1}_drz_sci.fits'.format(self.name, self.label),
                                      xsh=out_shift[0], ysh=out_shift[1],
                                      rot=out_rot, scale=out_scale,
                                      wcsname=ref_name, force=True,
                                      reusename=True, verbose=False,
                                      sciext='PRIMARY')
            
            self.hdu[i] = pyfits.open(tmpfile)[0]
        
        self.wcs = [pywcs.WCS(im.header) for im in self.hdu]
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
            tab[k.lower()] = np.array([h.header[k] for h in self.hdu])
        
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
                outfile = self.files[i].replace(mips_ext, ext)
            
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
                           
if __name__ == '__main__':
    psf_coords = (340.8852003, -9.604230278, get_bcd_psf)
    
    psf_coords = list(np.cast[float](ds9.get('pan fk5').split()))+[get_bcd_psf]
    
    kwargs = dict(wcslist=wcslist, driz_scale=driz_scale, theta=0, kernel=kernel, out_hdu=out_hdu, pad=pad, psf_coords=psf_coords)
    res = aors[aor].drizzle_output(pedestal=aors[aor].pedestal, **kwargs)
    
    #aors[aor].drz = aors[aor].drizzle_simple(wcslist=wcslist, driz_scale=driz_scale, theta=0, kernel=kernel, out_hdu=out_hdu, pad=pad, psf_coords=psf_coords)

    sci, wht, ctx, head, w = res #aors[aor].drz
    ds9.view(sci*1000, header=head); ds9.set('pan to {0} {1} fk5'.format(psf_coords[0], psf_coords[1]))

    root_i = '{0}-{1}-{2}'.format(output_root, aor, channel)
    pyfits.writeto('{0}_psf_sci.fits'.format(root_i), data=sci*100, header=head, overwrite=True)
    pyfits.writeto('{0}_psf_wht.fits'.format(root_i), data=wht, header=head, overwrite=True)
        
def mosaic_psf(output_root='irac', channel=1, pix=0.5, target_pix=0.1, pixfrac=0.2, kernel='square', aors={}, size=18, subtract_background=True, segmentation_mask=False, theta=0, native_orientation=False, instrument='irac', max_R=3.5):
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
        gaia = utils.read_catalog('{0}-{1}_gaia.radec'.format(aor, label))
        idx, dr = gaia.match_to_catalog_sky(cat)
        has_gaia = dr < 1*u.arcsec
        dd = gaia['dec'][idx] - cat['dec']
        dr = gaia['ra'][idx] - cat['ra']
        cat['rax'] = cat['ra'] + np.median(dr[has_gaia])
        cat['decx'] = cat['dec'] + np.median(dd[has_gaia])
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
            
            try:
                ds9.view(sci_sum/wht_sum*1000)
            except:
                pass
        
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

class MipsPSF(object):
    def __init__(self, scale=0.1, rmax=14):
        import grizli
        
        self.im = pyfits.open('/Users/gbrammer/Research/grizli/CONF/mips_24_100K.fits')
        self.data = self.im[0].data / self.im[0].data.sum()
        self.native_scale = 0.5 #0.498
        
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
        
        
class IracPSF(object):
    """
    Tools for generating an IRAC PSF at an arbitrary location in a mosaic
    """
    def __init__(self, ch=1, scale=0.1, verbose=True, aor_list=None, avg_psf=None):
        self.psf_data = {}
        self.psf_arrays = {}
        self.ch = ch
        self.scale = scale
        self.avg_psf = avg_psf

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
    
        psf_data = {}
        for file in files:
            aor = file.split('-ch')[0]
            if aor_list is not None:
                if aor not in aor_list:
                    continue

            log = utils.read_catalog(file)
                    
            psf_file = '{0}-ch{1}-{2:.1f}.psf.fits'.format(aor, ch, scale)
            if verbose:
                print('Read PSF data {0} / {1}'.format(file, psf_file))
                
            psf = pyfits.open(psf_file)
            psf_mask = (psf[0].data != 0)
            footprints = []
            for i, coo in enumerate(log['corners']):
                footprints.append(Path(coo))
            
            full_path = Path.make_compound_path(*footprints)
            hull = ConvexHull(full_path.vertices)
            full_footprint = Path(full_path.vertices[hull.vertices,:])
            
            psf_data[aor] = {'log':log, 'psf_image':psf[0].data, 
                             'psf_mask':psf_mask, 'bcd_footprints':footprints, 
                             'footprint': full_footprint}
    
        if avg_psf is None:
            img = np.array([psf_data[k]['psf_image'].flatten() for k in psf_data])
            msk = np.array([psf_data[k]['psf_mask'].flatten() for k in psf_data])
            psf_shape = psf[0].data.shape
        else:
            print('Use rotated `avg_psf`.')
            img = []
            msk = []
            for k in psf_data:
                theta = np.mean(psf_data[k]['log']['theta'])
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
    
    def evaluate_psf(self, ra=228.7540568, dec=-15.3806666, min_count=5, clip_negative=True, transform=None, warp_args={'order':3, 'mode':'constant', 'cval':0.}):
        """
        Evaluate weighted PSF at a given location using saved PSFs for each AOR
        
        transform: `~skimage.transform.SimilarityTransform`
        
        """
        import numpy as np
        from skimage.transform import warp
        
        exptime = np.zeros(self.N)
        count = np.zeros(self.N, dtype=int)
        
        point = (ra, dec)
        for i, aor in enumerate(self.psf_data):
            if not self.psf_data[aor]['footprint'].contains_point(point):
                continue
                
            for j, pat in enumerate(self.psf_data[aor]['bcd_footprints']):
                if pat.contains_point(point):
                    exptime[i] += self.psf_data[aor]['log']['exptime'][j]
                    count[i] += 1
        
        count_mask = (count > min_count)      
        if count_mask.sum() == 0:
            return np.zeros(self.psf_arrays['shape']), exptime, count
              
        expwht = exptime[count_mask]
        #num = (self.psf_arrays['image']*self.psf_arrays['mask'])
        num = self.psf_arrays['masked'][count_mask].T.dot(expwht)
        den = (self.psf_arrays['mask'][count_mask]).T.dot(expwht)
        
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
            
def warp_image(transform, image, warp_args={'order': 3, 'mode': 'constant', 'cval': 0.0}, center=None):
    
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
        cbcd = pyfits.open(wcsfile.replace('_wcs.fits','_ebcd.fits'))
        cbunc = pyfits.open(wcsfile.replace('_wcs.fits','_ebunc.fits'))
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
                
        
        