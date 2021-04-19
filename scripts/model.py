
def test_model():
    import numpy as np
    import grizli.ds9
    
    from importlib import reload
    import golfir.model
    reload(golfir.model)
    
    ds9 = grizli.ds9.DS9()
    
    bright = [12,16]
    bright = [18,18]
    bright_args = dict(any_limit=bright[0], point_limit=bright[1], point_flux_radius=3.5, bright_ids=None, bright_sn=7)
    mag_limit = 24
    
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


#
def fit_point_source(t0, poly_order=5):
    import grizli.utils
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
    ds9.view(_psf, header=grizli.utils.get_wcs_slice_header(irac_wcs, islx, isly))
    
    ds9.frame(18)
    ds9.view(irac_im.data[isly, islx], header=grizli.utils.get_wcs_slice_header(irac_wcs, islx, isly))

    psf_model = irac_im.data*0
    psf_model[isly, islx] += _psf

    ds9.frame(20)
    aa = 1
    ds9.view(irac_im.data-psf_model*aa, header=irac_im.header)
    ds9.set('pan to {0} {1} fk5'.format(rd_pan[0], rd_pan[1]))
    
    return _psf


def alma_source():
    """
    Model a source detected in alma
    """
    import grizli.utils
    
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
        
        phot = grizli.utils.read_catalog('{0}irac_phot_apcorr.fits'.format(root))

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
        
        alm = grizli.utils.read_catalog('alma.info', format='ascii')
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
            efnu.append(grizli.utils.nmad(alma[0].data[msk])*1.e6)
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
        
        tab = grizli.utils.GTable()
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
                
        jname = grizli.utils.radec_to_targname(ra=rg, dec=dg, round_arcsec=(0.01, 0.01*15), precision=2, targstr='j{rah}{ram}{ras}.{rass}{sign}{ded}{dem}{des}.{dess}', header=None)
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