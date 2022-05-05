"""
Full script for downloading and processing CHArGE fields
"""

import os
import glob
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path

from skimage.morphology import binary_dilation

import astropy.io.fits as pyfits
import astropy.wcs as pywcs
import astropy.units as u

try:
    import drizzlepac
    from drizzlepac.astrodrizzle import ablot
except:
    print("(golfir.pipeline) Warning: failed to import drizzlepac")
    
#from grizli import utils

def irac_mosaics(root='j000308m3303', home='/GrizliImaging/', pixfrac=0.2, kernel='square', initial_pix=1.0, final_pix=0.5, pulldown_mag=15.2, sync_xbcd=True, skip_fetch=False, radec=None, mosaic_pad=2.5, drizzle_ref_file='', run_alignment=True, assume_close=True, bucket='grizli-v1', aor_query='r*', mips_ext='[_e]bcd.fits', channels=['ch1','ch2','ch3','ch4','mips1'], drz_query='r*', sync_results=True, ref_seg=None, min_frame={'irac':5, 'mips':1.0}, med_max_size=500e6, stop_at='', make_psf=True, **kwargs):
    """
    stop_at: preprocess, make_compact
    
    """
    
    from grizli import utils

    from . import irac
    from .utils import get_wcslist, fetch_irac
    
    PATH = os.path.join(home, root)
    try:
        os.mkdir(PATH)
    except:
        pass

    os.chdir(PATH)
        
    if not skip_fetch:
        # Fetch IRAC bcds
        if not os.path.exists(f'{root}_ipac.fits'):
            os.system(f'wget https://s3.amazonaws.com/{bucket}/IRAC/{root}_ipac.fits')
    
        res = fetch_irac(root=root, path='./', channels=channels)
        
        if res in [False, None]:
            # Nothing to do
            make_html(root, bucket=bucket)

            print(f'### Done: \n https://s3.amazonaws.com/{bucket}/Pipeline/{root}/IRAC/{root}.irac.html')

            utils.log_comment(f'/tmp/{root}.success', 'Done!', 
                              verbose=True, show_date=True)
            return True
            
    # Sync CHArGE HST images
    os.system(f'aws s3 sync s3://{bucket}/Pipeline/{root}/Prep/ ./ '
              f' --exclude "*" --include "{root}*seg.fits*"'
              f' --include "{root}-ir_drz*fits*"'
              f' --include "{root}*psf.fits*"'
              f' --include "{root}-f[01]*_drz*fits.gz"'
              f' --include "{root}*phot.fits"')
    
    # Drizzle properties of the preliminary mosaic
    #pixfrac, pix, kernel = 0.2, 1.0, 'square'       
    
    # Define an output WCS aligned in pixel phase to the HST mosaic ()

    if not os.path.exists('ref_hdu.fits'):
        wcslist = get_wcslist(skip=-500)
        out_hdu = utils.make_maximal_wcs(wcslist, pixel_scale=initial_pix, theta=0, pad=5, get_hdu=True, verbose=True)

        # Make sure pixels align
        ref_file = glob.glob('{0}-f[01]*_drz_sci.fits*'.format(root))
        if len(ref_file) == 0:
            os.system(f'aws s3 sync s3://{bucket}/Pipeline/{root}/Prep/ ./ '
                      f' --exclude "*"'
                      f' --include "{root}-f[678]*_dr*fits.gz"')
            
            ref_file = glob.glob('{0}-f[678]*_dr*_sci.fits*'.format(root))
        
        ref_file = ref_file[-1]

        print(f'\nHST reference image: {ref_file}\n')

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

        out_hdu.writeto('ref_hdu.fits', output_verify='Fix')

    else:
        out_hdu = pyfits.open('ref_hdu.fits')[1]
    
    ########
    
    files = []
    for ch in channels:
        if 'mips' in ch:
            mc = ch.replace('mips','ch')
            files += glob.glob(f'{aor_query}/{mc}/bcd/SPITZER_M*{mips_ext}')
            files += glob.glob(f'{aor_query}/{mc}/bcd/SPITZER_M*xbcd.fits.gz')
        else:
            files += glob.glob(f'{aor_query}/{ch}/bcd/SPITZER_I*cbcd.fits')
            files += glob.glob(f'{aor_query}/{ch}/bcd/SPITZER_I*xbcd.fits.gz')
            
    files.sort()

    roots = np.array([file.split('/')[0] for file in files])
    with_channels = np.array([file.split('_')[1] for file in files])
    all_roots = np.array(['{0}-{1}'.format(r, c.replace('I','ch').replace('M', 'mips')) for r, c in zip(roots, with_channels)])

    tab = {'aor':[], 'N':[], 'channel':[]}
    for r in np.unique(all_roots):
        tab['aor'].append(r.split('-')[0])
        tab['N'].append((all_roots == r).sum())
        tab['channel'].append(r.split('-')[1])

    aors = utils.GTable(tab)
    print(aors)
    
    ########
    SKIP = True          # Don't regenerate finished files
    delete_group = False # Delete intermediate products from memory
    zip_outputs = False    # GZip intermediate products

    aors_ch = {}
    
    ########
    # Process mosaics by AOR
    # Process in groups, helps for fields like HFF with dozens/hundreds of AORs!
    for ch in channels:
            
        aor = aors[(aors['channel'] == ch) & (aors['N'] > 5)]
        if len(aor) == 0:
            continue

        #aors_ch[ch] = []

        if ch in ['ch1','ch2']:
            NPER, instrument = 500, 'irac'
        if ch in ['ch3','ch4']:
            NPER, instrument = 500, 'irac'
        elif ch in ['mips1']:
            NPER, instrument = 400, 'mips'
        
        min_frametime = min_frame[instrument]
        
        nsort = np.cumsum(aor['N']/NPER)
        NGROUP = int(np.ceil(nsort.max()))

        count = 0

        for g in range(NGROUP):
            root_i = root+'-{0:02d}'.format(g)

            gsel = (nsort > g) & (nsort <= g+1)
            aor_ids = list(aor['aor'][gsel])
            print('{0}-{1}   N_AOR = {2:>2d}  N_EXP = {3:>4d}'.format(root_i, ch,  len(aor_ids), aor['N'][gsel].sum()))
            count += gsel.sum()

            files = glob.glob('{0}-{1}*'.format(root_i, ch))
            if (len(files) > 0) & (SKIP): 
                print('Skip {0}-{1}'.format(root_i, ch))
                continue
            
            with open('{0}-{1}.log'.format(root_i, ch),'w') as fp:
                fp.write(time.ctime())
                
            # Do internal alignment to GAIA.  
            # Otherwise, set `radec` to the name of a file that has two columns with 
            # reference ra/dec.
            #radec = None 

            # Pipeline
            if instrument == 'mips':
                aors_ch[ch] = irac.process_all(channel=ch.replace('mips','ch'), output_root=root_i, driz_scale=initial_pix, kernel=kernel, pixfrac=pixfrac, wcslist=None, pad=0, out_hdu=out_hdu, aor_ids=aor_ids, flat_background=False, two_pass=True, min_frametime=min_frametime, instrument=instrument, align_threshold=0.15, radec=radec, run_alignment=False, mips_ext=mips_ext, ref_seg=ref_seg, global_mask=root+'_mask.reg')
            else:
                aors_ch[ch] = irac.process_all(channel=ch, output_root=root_i, driz_scale=initial_pix, kernel=kernel, pixfrac=pixfrac, wcslist=None, pad=0, out_hdu=out_hdu, aor_ids=aor_ids, flat_background=False, two_pass=True, min_frametime=min_frametime, instrument=instrument, radec=radec, run_alignment=run_alignment, assume_close=assume_close, ref_seg=ref_seg, global_mask=root+'_mask.reg', med_max_size=med_max_size)

            if len(aors_ch[ch]) == 0:
                continue

            # PSFs
            plt.ioff()

            if (instrument != 'mips') & make_psf:
                ch_num = int(ch[-1])
                segmask=True

                # psf_size=20
                # for p in [0.1, final_pix]:
                #     irac.mosaic_psf(output_root=root_i, target_pix=p, channel=ch_num, aors=aors_ch[ch], kernel=kernel, pixfrac=pixfrac, size=psf_size, native_orientation=False, instrument=instrument, subtract_background=False, segmentation_mask=segmask, max_R=10)
                #     plt.close('all')

                psf_size=30
                p = 0.1
                irac.mosaic_psf(output_root=root_i, target_pix=p, channel=ch_num, aors=aors_ch[ch], kernel=kernel, pixfrac=pixfrac, size=psf_size, native_orientation=True, subtract_background=False, segmentation_mask=segmask, max_R=10)

                plt.close('all')

            if delete_group:
                del(aors_ch[ch])

            print('Done {0}-{1}, gzip products'.format(root_i, ch))

            if zip_outputs:
                os.system('gzip {0}*-{1}_drz*fits'.format(root_i, ch))
        
        # PSFs
        if (instrument != 'mips') & make_psf:
            # Average PSF
            p = 0.1
            files = glob.glob('*{0}-{1:.1f}*psfr.fits'.format(ch, p))
            if len(files) == 0:
                continue
                
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
                
                im.close()
                
            avg = avg/wht
            avg[wht == 0] = 0

            # Window
            try:    
                from photutils import (HanningWindow, TukeyWindow, 
                                        CosineBellWindow, 
                                        SplitCosineBellWindow, 
                                        TopHatWindow)
            except:
                from photutils.psf.matching import (HanningWindow, 
                                        TukeyWindow, 
                                        CosineBellWindow, 
                                        SplitCosineBellWindow, 
                                        TopHatWindow)

            coswindow = CosineBellWindow(alpha=1)
            avg *= coswindow(avg.shape)**0.05
            avg /= avg.sum()

            pyfits.writeto('{0}-{1}-{2:0.1f}.psfr_avg.fits'.format(root, ch, p), data=avg, header=im[0].header, overwrite=True)
    
    ####
    ## Show the initial product
    plt.ioff()
    for i in range(10):
        files = glob.glob(f'{root}-{i:02d}-ch*sci.fits')
        if len(files) > 0:
            break
            
    files.sort()
    
    if len(files) == 1:
        subs = 1,1
        fs = [7,7]
    elif len(files) == 2:
        subs = 1,2
        fs = [14,7]
    elif len(files) == 3:
        subs = 2,2
        fs = [14,14]
    else:
        subs = 2,2
        fs = [14,14]
        
    fig = plt.figure(figsize=fs)
    for i, file in enumerate(files[:4]):
        im = pyfits.open(file)
        print('{0} {1} {2:.1f} s'.format(file, im[0].header['FILTER'], im[0].header['EXPTIME']))
        ax = fig.add_subplot(subs[0], subs[1], 1+i)
        ax.imshow(im[0].data, vmin=-0.1, vmax=1, cmap='gray_r', origin='lower')
        ax.text(0.05, 0.95, file, ha='left', va='top', color='k', 
                transform=ax.transAxes)
        
        im.close()
        
    if len(files) > 1:
        fig.axes[1].set_yticklabels([])
    
    if len(files) > 2:
        fig.axes[0].set_xticklabels([])
        fig.axes[1].set_xticklabels([])
    
    if len(files) > 3:
        fig.axes[3].set_yticklabels([])
        
    fig.tight_layout(pad=0.5)
    fig.savefig(f'{root}.init.png')
    plt.close('all')
    
    if stop_at == 'preprocess':
        return True
        
    #######
    # Make more compact individual exposures and clean directories
    wfiles = []
    for ch in channels:
        if 'mips' in ch:
            chq = ch.replace('mips','ch')
            wfiles += glob.glob(f'{aor_query}/{chq}/bcd/SPITZER_M*wcs.fits')
        else:
            wfiles += glob.glob(f'{aor_query}/{ch}/bcd/SPITZER_I*wcs.fits')

    #wfiles = glob.glob('r*/*/bcd/*_I[1-4]_*wcs.fits')
    #wfiles += glob.glob('r*/*/bcd/*_M[1-4]_*wcs.fits')
    wfiles.sort()

    for wcsfile in wfiles:
        outfile = wcsfile.replace('_wcs.fits', '_xbcd.fits.gz')
        if os.path.exists(outfile):
            print(outfile)
        else:
            irac.combine_products(wcsfile)
            print('Run: ', outfile)

        if os.path.exists(outfile):
            remove_files = glob.glob('{0}*fits'.format(wcsfile.split('_wcs')[0]))
            for f in remove_files:
                print('   rm ', f)
                os.remove(f)
 
    if stop_at == 'make_compact':
        return True
                                   
    #############
    # Drizzle final mosaics
    # Make final mosaic a bit bigger than the HST image
    pad = mosaic_pad

    # Pixel scale of final mosaic.
    # Don't make too small if not many dithers available as in this example.
    # But for well-sampled mosaics like RELICS / HFF, can push this to perhaps 0.3" / pix
    pixscale = final_pix #0.5

    # Again, if have many dithers maybe can use more aggressive drizzle parameters,
    # like a 'point' kernel or smaller pixfrac (a 'point' kernel is pixfrac=0)
    #kernel, pixfrac = 'square', 0.2

    # Correction for bad columns near bright stars
    #pulldown_mag = 15.2 

    ##############
    # Dilation for CR rejection
    dil = np.ones((3,3))
    driz_cr = [7, 4]
    blot_interp = 'poly5'
    bright_fmax = 0.5
    
    ### Drizzle
    for ch in channels: #[:2]:
        ###########
        # Files and reference image for extra CR rejection
        if ch == 'mips1':
            files = glob.glob('{0}/ch1/bcd/SPITZER_M1_*xbcd.fits*'.format(drz_query, ch))
            files.sort()
            pulldown_mag = -10
            pixscale = 1.
            kernel = 'point'
        else:
            files = glob.glob('{0}/{1}/bcd/*_I?_*xbcd.fits*'.format(drz_query, ch))
            files.sort()

        #ref = pyfits.open('{0}-00-{1}_drz_sci.fits'.format(root, ch))
        #ref_data = ref[0].data.astype(np.float32)

        ref_files = glob.glob(f'{root}-??-{ch}*sci.fits')
        if len(ref_files) == 0:
            continue

        num = None
        for ref_file in ref_files:
            ref = pyfits.open(ref_file)
            wht = pyfits.open(ref_file.replace('_sci.fits', '_wht.fits'))
            if num is None:
                num = ref[0].data*wht[0].data
                den = wht[0].data
            else:
                num += ref[0].data*wht[0].data
                den += wht[0].data

        ref_data = (num/den).astype(np.float32)
        ref_data[den <= 0] = 0

        ref_wcs = pywcs.WCS(ref[0].header, relax=True) 
        ref_wcs.pscale = utils.get_wcs_pscale(ref_wcs) 
        if (not hasattr(ref_wcs, '_naxis1')) & hasattr(ref_wcs, '_naxis'):
            ref_wcs._naxis1, ref_wcs._naxis2 = ref_wcs._naxis

        ##############
        # Output WCS based on HST footprint
        if drizzle_ref_file == '':
            try:
                hst_im = pyfits.open(glob.glob('{0}-f[01]*_drz_sci.fits*'.format(root))[-1])
            except:
                hst_im = pyfits.open(glob.glob('{0}-f[578]*_dr*sci.fits*'.format(root))[-1])
            
    
            hst_wcs = pywcs.WCS(hst_im[0])
            hst_wcs.pscale = utils.get_wcs_pscale(hst_wcs) 

            try:
                size = (np.round(np.array([hst_wcs._naxis1, hst_wcs._naxis2])*hst_wcs.pscale*pad/pixscale)*pixscale)
            except:
                size = (np.round(np.array([hst_wcs._naxis[0], hst_wcs._naxis[1]])*hst_wcs.pscale*pad/pixscale)*pixscale)
            
            hst_rd = hst_wcs.calc_footprint().mean(axis=0)
            _x = utils.make_wcsheader(ra=hst_rd[0], dec=hst_rd[1],
                                      size=size, 
                                      pixscale=pixscale, 
                                      get_hdu=False, theta=0)
            
            out_header, out_wcs = _x
        else:
            driz_ref_im = pyfits.open(drizzle_ref_file)
            out_wcs = pywcs.WCS(driz_ref_im[0].header, relax=True)
            out_wcs.pscale = utils.get_wcs_pscale(out_wcs) 
            
            out_header = utils.to_header(out_wcs)
        
        if (not hasattr(out_wcs, '_naxis1')) & hasattr(out_wcs, '_naxis'):
            out_wcs._naxis1, out_wcs._naxis2 = out_wcs._naxis
            
        ##############
        # Bright stars for pulldown correction
        cat_file = glob.glob(f'{root}-[0-9][0-9]-{ch}.cat.fits')[0]
        ph = utils.read_catalog(cat_file) 
        bright = (ph['mag_auto'] < pulldown_mag) # & (ph['flux_radius'] < 3)
        ph = ph[bright]

        ##############
        # Now do the drizzling
        yp, xp = np.indices((256, 256))
        orig_files = []

        out_header['DRIZ_CR0'] = driz_cr[0]
        out_header['DRIZ_CR1'] = driz_cr[1]
        out_header['KERNEL'] = kernel
        out_header['PIXFRAC'] = pixfrac
        out_header['NDRIZIM'] = 0
        out_header['EXPTIME'] = 0
        out_header['BUNIT'] = 'microJy'
        out_header['FILTER'] = ch

        med_root = 'xxx'
        N = len(files)

        for i, file in enumerate(files):#[:100]):

            print('{0}/{1} {2}'.format(i, N, file))

            if file in orig_files:
                continue

            im = pyfits.open(file)
            ivar = 1/im['CBUNC'].data**2    
            msk = (~np.isfinite(ivar)) | (~np.isfinite(im['CBCD'].data))
            im['CBCD'].data[msk] = 0
            ivar[msk] = 0

            wcs = pywcs.WCS(im['WCS'].header, relax=True)
            wcs.pscale = utils.get_wcs_pscale(wcs)
            if (not hasattr(wcs, '_naxis1')) & hasattr(wcs, '_naxis'):
                wcs._naxis1, wcs._naxis2 = wcs._naxis
            
            fp = Path(wcs.calc_footprint())

            med_root_i = im.filename().split('/')[0]
            if med_root != med_root_i:
                print('\n Read {0}-{1}_med.fits \n'.format(med_root_i, ch))
                med = pyfits.open('{0}-{1}_med.fits'.format(med_root_i, ch))
                med_data = med[0].data.astype(np.float32)
                med_root = med_root_i
                med.close()
                
                try:
                    gaia_rd = utils.read_catalog('{0}-{1}_gaia.radec'.format(med_root_i, ch))
                    ii, rr = gaia_rd.match_to_catalog_sky(ph)
                    gaia_rd = gaia_rd[ii][rr.value < 2]
                    gaia_pts = np.array([gaia_rd['ra'].data, 
                                         gaia_rd['dec'].data]).T
                except:
                    gaia_rd = []

            #data = im['CBCD'].data - aor_med[0].data

            # Change output units to uJy / pix
            if ch == 'mips1':
                # un = 1*u.MJy/u.sr
                # #to_ujy_px = un.to(u.uJy/u.arcsec**2).value*(out_wcs.pscale**2)
                # to_ujy_px = un.to(u.uJy/u.arcsec**2).value*(native_scale**2)
                to_ujy_px = 146.902690
            else:
                # native_scale = 1.223
                # un = 1*u.MJy/u.sr
                # #to_ujy_px = un.to(u.uJy/u.arcsec**2).value*(out_wcs.pscale**2)
                # to_ujy_px = un.to(u.uJy/u.arcsec**2).value*(native_scale**2)
                to_ujy_px = 35.17517196810

            blot_data = ablot.do_blot(ref_data, ref_wcs, wcs, 1, coeffs=True, 
                                      interp=blot_interp, 
                                      sinscl=1.0, stepsize=10, 
                                      wcsmap=None)/to_ujy_px

            # mask for bright stars
            eblot = 1-np.clip(blot_data, 0, bright_fmax)/bright_fmax

            # Initial CR
            clean = im[0].data - med_data - im['WCS'].header['PEDESTAL']
            dq = (clean - blot_data)*np.sqrt(ivar)*eblot > driz_cr[0]

            # Adjacent CRs
            dq_dil = binary_dilation(dq, selem=dil)
            dq |= ((clean - blot_data)*np.sqrt(ivar)*eblot > driz_cr[1]) & (dq_dil)

            # Very negative pixels
            dq |= clean*np.sqrt(ivar) < -4

            original_dq = im['WCS'].data - (im['WCS'].data & 1)
            dq |= original_dq > 0

            # Pulldown correction for bright stars
            if len(gaia_rd) > 0:       
                mat = fp.contains_points(gaia_pts) 
                if mat.sum() > 0:
                    xg, yg = wcs.all_world2pix(gaia_rd['ra'][mat], gaia_rd['dec'][mat], 0)
                    sh = dq.shape
                    mat = (xg > 0) & (xg < sh[1]) & (yg > 0) & (yg < sh[0])
                    if mat.sum() > 0:
                        for xi, yi in zip(xg[mat], yg[mat]):
                            dq |= (np.abs(xp-xi) < 2) & (np.abs(yp-yi) > 10)

            if i == 0:
                res = utils.drizzle_array_groups([clean], [ivar*(dq == 0)], [wcs], outputwcs=out_wcs, kernel=kernel, pixfrac=pixfrac, data=None, verbose=False)
                # Copy header keywords
                wcs_header = utils.to_header(wcs)
                for k in im[0].header:
                    if (k not in ['', 'HISTORY', 'COMMENT']) & (k not in out_header) & (k not in wcs_header):
                        out_header[k] = im[0].header[k]

            else:
                _ = utils.drizzle_array_groups([clean], [ivar*(dq == 0)], [wcs], outputwcs=out_wcs, kernel=kernel, pixfrac=pixfrac, data=res[:3], verbose=False)

            out_header['NDRIZIM'] += 1
            out_header['EXPTIME'] += im[0].header['EXPTIME']
            
            im.close()
            
        # Pixel scale factor for weights
        wht_scale = (out_wcs.pscale/wcs.pscale)**-4

        # Write final images
        pyfits.writeto('{0}-{1}_drz_sci.fits'.format(root, ch), data=res[0]*to_ujy_px, header=out_header, 
                       output_verify='fix', overwrite=True)
        pyfits.writeto('{0}-{1}_drz_wht.fits'.format(root, ch), data=res[1]*wht_scale/to_ujy_px**2, 
                       header=out_header, output_verify='fix', overwrite=True)
    
    ##########
    ## Show the final drizzled images
    plt.ioff()
    files = glob.glob(f'{root}-ch*sci.fits')
    files.sort()
    
    if len(files) == 1:
        subs = 1,1
        fs = [7,7]
    elif len(files) == 2:
        subs = 1,2
        fs = [14,7]
    elif len(files) == 3:
        subs = 2,2
        fs = [14,14]
    else:
        subs = 2,2
        fs = [14,14]
        
    fig = plt.figure(figsize=fs)
    for i, file in enumerate(files[:4]):
        im = pyfits.open(file)
        print('{0} {1} {2:.1f} s'.format(file, im[0].header['FILTER'], im[0].header['EXPTIME']))
        ax = fig.add_subplot(subs[0], subs[1], 1+i)
        scl = (final_pix/initial_pix)**2
        ax.imshow(im[0].data, vmin=-0.1*scl, vmax=1*scl, cmap='gray_r', origin='lower')
        ax.text(0.05, 0.95, file, ha='left', va='top', color='k', 
                transform=ax.transAxes)
        
        im.close()
        
    if len(files) > 1:
        fig.axes[1].set_yticklabels([])
    
    if len(files) > 2:
        fig.axes[0].set_xticklabels([])
        fig.axes[1].set_xticklabels([])
    
    if len(files) > 3:
        fig.axes[3].set_yticklabels([])
        
    fig.tight_layout(pad=0.5)
    fig.savefig(f'{root}.final.png')
    plt.close('all')
    
    if sync_results:
        print('gzip mosaics')
        os.system(f'gzip -f {root}-ch*_drz*fits {root}-mips*_drz*fits')
    
        ######## Sync
        ## Sync
        print(f's3://{bucket}/Pipeline/{root}/IRAC/')
    
        make_html(root, bucket=bucket)
    
        os.system(f'aws s3 sync ./ s3://{bucket}/Pipeline/{root}/IRAC/'
                  f' --exclude "*" --include "{root}-ch*drz*fits*"'
                  f' --include "{root}-mips*drz*fits*"'
                  f' --include "{root}.*png"'
                  ' --include "*-ch*psf*" --include "*log.fits"' 
                  ' --include "*wcs.[lp]*"'
                  ' --include "*html" --include "*fail*"'
                  ' --acl public-read')
    
        if sync_xbcd:
            aor_files = glob.glob('r*-ch*med.fits')
            for aor_file in aor_files:
                aor = aor_file.split('-ch')[0]
                os.system(f'aws s3 sync ./{aor}/ s3://{bucket}/IRAC/AORS/{aor}/ --exclude "*" --include "ch*/bcd/*xbcd.fits.gz" --acl public-read')
                os.system(f'aws s3 cp {aor_file} s3://{bucket}/IRAC/AORS/ --acl public-read')
                
    msg = f'### Done: \n    https://s3.amazonaws.com/{bucket}/Pipeline/{root}/IRAC/{root}.irac.html'
       
    utils.log_comment(f'/tmp/{root}.success', msg, verbose=True, show_date=True)
    
def make_html(root, bucket='grizli-v1'):
    import time
    
    im = pyfits.open(glob.glob(f'{root}-ch*sci.fits*')[0])
    ra = im[0].header['CRVAL1']
    dec = im[0].header['CRVAL2']
    
    radius = 20. # arcmin

    URL = "https://sha.ipac.caltech.edu/applications/Spitzer/SHA/#id=SearchByPosition&"
    URL += f"RequestClass=ServerRequest&DoSearch=true&SearchByPosition.field.radius={radius/60:.5f}"
    URL += f"&UserTargetWorldPt={ra};{dec};EQ_J2000&SimpleTargetPanel.field.resolvedBy=nedthensimbad&"
    URL += "MoreOptions.field.prodtype=aor,pbcd,bcd&shortDesc=Position&isBookmarkAble=true&isDrillDownRoot=true&"
    URL += "isSearchResult=true"
    
    
    html = f"""
<h3> {root} IRAC ({time.ctime()})</h3>

<p>
<a href="https://s3.amazonaws.com/{bucket}/Pipeline/{root}/Prep/{root}.summary.html">CHArGE HST</a>
<p> SHA <a href="{URL}">query</a>
<p>
<a href="https://s3.amazonaws.com/{bucket}/IRAC/{root}_ipac.png"><img src="https://s3.amazonaws.com/{bucket}/IRAC/{root}_ipac.png" height=400></a>

<p>
<a href="{root}.init.png"><img src="{root}.init.png" width=800></a>
<br>
<a href="{root}.final.png"><img src="{root}.final.png" width=800></a>

<pre>
"""
    
    groups = [(f'{root}-[cm][hi]*drz*', 'Mosaics'), ('r*log.fits', 'Log'), ('*fail*', 'Failed'), ('r*psf.*', 'PSFs'), ('r*-ch2*psf.*', 'CH2 PSFs'), ('r*-ch3*psf.*', 'CH3 PSFs'), ('r*-ch4*psf.*', 'CH4 PSFs')][:-3]
    for g in groups:
        files = glob.glob(g[0])
        if len(files) == 0:
            continue
        files.sort()
        html += '\n####### {0}\n'.format(g[1])
        for file in files:
            html += f'<a href="{file}">{file}</a>\n'
    
    html += '</pre>\n'
    
    for ch in ['ch1', 'ch2', 'ch3', 'ch4']:
        files = glob.glob(f'*-{ch}*psfr.fits')
        files.sort()
        if len(files) == 0:
            continue
        
        html += '<p>\n'
        for file in files:
              html += '<a href="{0}"><img src={1} width=200></a>\n'.format(file, file.replace('.fits','.png'))
    
    html += '<h4>Alignment</h4><table>'
    files = glob.glob('*wcs.log')
    files.sort()
    for file in files:
        html += '<tr><td> <a href="{0}"><img src={0} width=150></a>\n<pre></td><td> <pre>'.format(file.replace('.log','.png'))
        
        lines = open(file).readlines()
        html += ''.join(lines)
        html += '</pre></td></tr>\n'
       
    html += '</table>'
    
    fp = open(f'{root}.irac.html','w')
    fp.write(html)
            
    fp.close()
    