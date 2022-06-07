import os
import glob
import time
import numpy as np
import matplotlib.pyplot as plt

from astropy.time import Time
import astropy.io.fits as pyfits
import astropy.wcs as pywcs

from skimage.morphology import binary_dilation

try:
    from tristars import match
except:
    print("(golfir.hawki) Warning: failed to import tristars")

try:
    from grizli import prep, utils
    #import grizli.ds9
except:
    print("(golfir.hawki) Warning: failed to import grizli")
    
#ds9 = grizli.ds9.DS9()

import traceback

# import astroquery.eso
# eso = astroquery.eso.Eso()
# if False:
#     eso.login() #reenter_password=True)

def full_query():
    """
    """
    MAXREC=5000
    query = """
    SELECT ob_id, target, avg(ra) as ra, avg(dec) as dec, 
           min(ra) minra, max(ra) as maxra,
           min(dec) as mindec, max(dec) as maxdec,
           count(exposure) as n, sum(exposure)/3600 as hours,
           prog_id,pi_coi 
    FROM dbo.raw
    WHERE (instrument='HAWKI') AND (filter_path='KS,OPEN')
    AND ra > 0 AND dec > -90
    AND tpl_nexp > 10 GROUP BY ob_id, target, prog_id, pi_coi
        """
    #
    qstr = query.replace("'","%27").replace(' ','+').replace('\n','+').replace('=','%3D')
    qstr = qstr.replace('>','%3E').replace('<','%3C')
    TAP_URL = "http://archive.eso.org/tap_obs/sync?REQUEST=doQuery&FORMAT=fits&LANG=ADQL&MAXREC={0}&QUERY={1}"
    SEND = TAP_URL.format(MAXREC, qstr)
    print(SEND)

    im = pyfits.open(SEND)
    hi = utils.GTable()
    for c in im[1].data.columns:
        if len(im[1].data[c.name]) > 0:
            hi[c.name] = im[1].data[c.name]
    
    hi['pi'] = [p.strip() for p in hi['pi_coi']]
    hi = hi[hi['pi'] != '']
    
    hi['pi'] = [p.strip().split()[0].strip('/').split('/')[0] for p in hi['pi_coi']]


def hawki_tap_query(rd=(325.6038899,-44.3295833), min_nexp=10, filter=" AND (filter_path='KS,OPEN')", extra=''):
    """
    Get from http://archive.eso.org/tap_obs
    """
    import astropy.io.fits as pyfits
    from grizli import utils
    import os
    import numpy as np
    
    coord = "AND ra > {0} AND ra < {1} AND dec > {2} AND dec < {3}".format(rd[0]-1.0, rd[0]+1.0, rd[1]-1.0, rd[1]+1.0) # UDS

    query = f"""SELECT * FROM dbo.raw
    WHERE (instrument='HAWKI') {filter}
    AND tpl_nexp > {min_nexp}
    {coord}
    {extra}
    """
    #query = "SELECT * FROM ivoa.ObsCore WHERE obs_collection='MUSE-DEEP' OR obs_collection='MUSE'"

    MAXREC=200000
    qstr = query.replace("'","%27").replace(' ','+').replace('\n','+').replace('=','%3D')
    qstr = qstr.replace('>','%3E').replace('<','%3C')
    TAP_URL = "http://archive.eso.org/tap_obs/sync?REQUEST=doQuery&FORMAT=fits&LANG=ADQL&MAXREC={0}&QUERY={1}"
    SEND = TAP_URL.format(MAXREC, qstr)
    print(SEND)

    im = pyfits.open(SEND)
    hi = utils.GTable()
    for c in im[1].data.columns:
        if len(im[1].data[c.name]) > 0:
            hi[c.name] = im[1].data[c.name]
    #
    #so = np.argsort(hi['ob_id']*100+hi['tpl_expno'])
    so = np.argsort(hi['exp_start'])
    hi = hi[so]
        
    field = utils.radec_to_targname(*rd)
                             
    if len(hi) > 0:
        print(f'{field}: {len(hi)} datasets')
        hi['pi_coi'] = [p.split('/')[0].strip() for p in hi['pi_coi']]
        hi.write(f'{field}_hawki.fits', overwrite=True)
    
    return hi
    
    path = f'{field}/RAW'
    for d in [field, f'{field}/RAW']:
        if not os.path.exists(d):
            os.mkdir(d)
    
    os.chdir(d)
    
    #hi['obs_collection'] = [o.strip() for o in hi['obs_collection']]
    for i, f in enumerate(hi['access_url']):
        local_file = os.path.basename(f)+'.fits'
        if os.path.exists(local_file):
            print(i, local_file)
        else:
            print(f'\n\n{i+1} / {len(hi)} : Fetch {local_file}\n')
            os.system(f'wget "{f}" -O {local_file}.Z -nv')
            os.system(f'gunzip {local_file}.Z')


def sip_avg():
    import glob
    from tqdm import tqdm
    from astropy.io.fits import Header
    
    files = glob.glob('ob1012480-20131025-005002-0*-1.cat.fits')

    chip = 1    
    files = glob.glob(f'*-{chip}.cat.fits')
    
    files.sort()
    roots = [f.split('.cat.fits')[0] for f in files]
    
    h = [fit_sip(root=root, degree=3, radec='../m4-gaia.radec') for root in tqdm(roots)]
    
    avgh = Header()
    for k in h[0]:
        avgh[k] = h[0][k]
        
    keys = {}
    for k in h[0]:
        if isinstance(h[0][k], float):
            keys[k] = []
                
    for i, hi in enumerate(h):
        print(i, hi['SIPRAMAD'], hi['SIPDEMAD'])
        for k in keys:
            keys[k].append(hi[k])
    
    for k in keys:
        stats = np.percentile(keys[k], [5, 16, 50, 84, 95])
        avgh[k] = stats[2]
        print(k, stats)
    
    lib = Header.fromtextfile(f'../libralato_hawki_chip{chip}.crpix.header')
    for k in list(avgh.keys()):
        if k not in lib:
            avgh.remove(k)
        else:
            print(k, avgh[k], lib[k])
    
    avgh['OEXTNAME'] = lib['OEXTNAME']
    avgh['COMMENT'] = 'SIP fit to M4 OB 657973'
    avgh.totextfile(f'../m4_hawki_chip{chip}.crpix.header', overwrite=True)
    avgh.totextfile(f'../libralato_hawki_chip{chip}.crpix.header', overwrite=True)


def fit_sip(root='ob1012480-20131025-005002-002-1', radec='../hst.radec', degree=3):
    """
    """     
    from grizli import jwst_utils, utils
    import astropy.io.fits as pyfits
    import astropy.wcs as pywcs
    from astropy.io.fits import Header
    from scipy.optimize import least_squares
    
    rd = utils.read_catalog(radec)
    im = pyfits.open(f'{root}.sci.fits')
    wcs = pywcs.WCS(im[0].header) 
    cat = utils.read_catalog(f'{root}.cat.fits')
    idx, dr = rd.match_to_catalog_sky(cat)
    hasm = dr.value < 0.25
    
    x, y = rd['ra'][idx][hasm], rd['dec'][idx][hasm]
    u, v = cat['x'][hasm], cat['y'][hasm]
    
    lsq_args = jwst_utils.LSQ_ARGS
    
    #degree = 3
    
    crpix = 1024.5, 1024.5
    crpix = im[0].header['CRPIX1'], im[0].header['CRPIX2']
    
    crval = np.squeeze(wcs.all_pix2world([crpix[0]], [crpix[1]], 1))
    
    header = Header()
    header['RADESYS'] = 'ICRS'
    header['CTYPE1'] = 'RA---TAN'
    header['CTYPE2'] = 'DEC--TAN'

    header['CUNIT1'] = header['CUNIT2'] = 'deg'

    header['CRPIX1'] = crpix[0]
    header['CRPIX2'] = crpix[1]

    header['CRVAL1'] = crval[0]
    header['CRVAL2'] = crval[1]

    cosd = np.cos(crval[1]/180*np.pi)
    cd = wcs.wcs.cd
    header['CD1_1'] = cd[0][0]
    header['CD1_2'] = cd[0][1]

    header['CD2_1'] = cd[1][0]
    header['CD2_2'] = cd[1][1]
    
    step = 32
    xmin = ymin = step
    
    sh = im[0].data.shape
    
    a_names = []
    b_names = []
    
    a_rand = []
    b_rand = []
    
    sip_step = []
    
    for i in range(degree+1):
        for j in range(degree+1):
            ext = '{0}_{1}'.format(i, j)
            if (i+j) > degree:
                continue

            if ext in ['0_0', '0_1', '1_0']:
                continue

            a_names.append('A_'+ext)
            b_names.append('B_'+ext)
            sip_step.append(1.e-3**(i+j))
            
    p0 = np.zeros(4+len(a_names)+len(b_names))
    p0[:4] += cd.flatten()
    
    args = (u.flatten(), v.flatten(), x.flatten(), y.flatten(), crval, crpix, 
            a_names, b_names, cd, 0)

    # Fit the SIP coeffs
    fit = least_squares(jwst_utils._objective_sip, p0, args=args, **lsq_args)

    # Get the results
    args = (u.flatten(), v.flatten(), x.flatten(), y.flatten(), crval, crpix, 
            a_names, b_names, cd, 1)

    cd_fit, a_coeff, b_coeff, ra_nmad, dec_nmad = jwst_utils._objective_sip(fit.x, *args)
    #
    for i in range(2):
        for j in range(2):
            header['CD{0}_{1}'.format(i+1, j+1)] = cd_fit[i, j]

    header['CTYPE1'] = 'RA---TAN-SIP'
    header['CTYPE2'] = 'DEC--TAN-SIP'
    
    header['NAXIS'] = 2
    #sh = datamodel.data.shape
    header['NAXIS1'] = sh[1]
    header['NAXIS2'] = sh[0]
    
    header['A_ORDER'] = degree
    for k in a_coeff:
        header[k] = a_coeff[k]

    header['B_ORDER'] = degree
    for k in b_coeff:
        header[k] = b_coeff[k]
    
    header['PIXSCALE'] = utils.get_wcs_pscale(header), 'Derived pixel scale'
    
    header['SIPSTATU'] = fit.status, 'least_squares result status'
    header['SIPCOST'] = fit.cost, 'least_squares result cost'
    header['SIPNFEV'] = fit.nfev, 'least_squares result nfev'
    header['SIPNJEV'] = fit.njev, 'least_squares result njev'
    header['SIPOPTIM'] = fit.optimality, 'least_squares result optimality'
    header['SIPSUCSS'] = fit.success, 'least_squares result success'
    header['SIPRAMAD'] = ra_nmad/header['PIXSCALE'], 'RA NMAD, pix'
    header['SIPDEMAD'] = dec_nmad/header['PIXSCALE'], 'Dec NMAD, pix'
    header['NMATCH'] = len(u)*1.
    
    wcs_new = pywcs.WCS(header)
    rd_new = wcs_new.all_pix2world(u, v, 0)
    c = utils.GTable()
    c['ra'], c['dec'] = rd_new
    id, dd, dx_new, dy_new = rd.match_to_catalog_sky(c, get_2d_offset=True)
    
    crval[0] += np.median(dx_new.value)/3600/cosd
    crval[1] += np.median(dy_new.value)/3600
    
    rd_old = wcs.all_pix2world(u, v, 0)
    c = utils.GTable()
    c['ra'], c['dec'] = rd_old
    id, dd, dx_old, dy_old = rd.match_to_catalog_sky(c, get_2d_offset=True)

    header['X_OLD'] = utils.nmad(dy_old.value)
    header['Y_OLD'] = utils.nmad(dy_old.value)
    header['X_NEW'] = utils.nmad(dy_new.value)
    header['Y_NEW'] = utils.nmad(dy_new.value)
    
    return header
    
    
def distortion_m4():
    """
    """
    from grizli import utils
    filters = ['Ks']
    min_nexp = 10
    
    rd = 245.8999645, -26.5252304
    
    kwargs = {}
    kwargs['column_filters'] = {}
    kwargs['column_filters']['tpl_nexp'] = [f'> {min_nexp}']
    kwargs['column_filters']['tpl_expno'] = [1]
    kwargs['column_filters']['ins_filt1_name'] = filters
    kwargs['column_filters']['dp_cat'] = ['SCIENCE']
    #kwargs['column_filters']['dp_tech'] = ['IMAGE']
    kwargs['column_filters']['pi_coi_name'] = 'PI_only'
    
    #kwargs['column_filters']['RA'] = [f'> {rd[0]-0.5}', f'< {rd[0]}+=0.5']
    
    kwargs['columns'] = ['tpl_nexp', 'tpl_expno', 'det_ndit', 'det_dit', 'det_ncorrs_name', 'obs_tmplno', 'det_ncorrs_name', 'prog_type', 'pi_coi']
    
    #res = eso.query_instrument('hawki', pi_coi_name='PI_only', **kwargs)
    #res['PI'] = [p.split('/')[0].strip() for p in res['PI/CoI']]
    
    kws = {}
    for k in kwargs:
        kws[k] = kwargs[k].copy()
    
    kws['column_filters'].pop('tpl_nexp')
    kws['column_filters'].pop('tpl_expno')
    
    kws['column_filters']['ob_id'] = [657973]
    
    coord = "AND ra > {0} AND ra < {1} AND dec > {2} AND dec < {3}".format(rd[0]-1.0, rd[0]+1.0, rd[1]-1.0, rd[1]+1.0) # UDS

    query = f"""SELECT * FROM dbo.raw
    WHERE (instrument='HAWKI') AND (filter_path='KS,OPEN')
    AND tpl_nexp > {min_nexp} AND ob_id = 657973
    {coord}
    """
    #query = "SELECT * FROM ivoa.ObsCore WHERE obs_collection='MUSE-DEEP' OR obs_collection='MUSE'"

    MAXREC=200000
    qstr = query.replace("'","%27").replace(' ','+').replace('\n','+').replace('=','%3D')
    qstr = qstr.replace('>','%3E').replace('<','%3C')
    TAP_URL = "http://archive.eso.org/tap_obs/sync?REQUEST=doQuery&FORMAT=fits&LANG=ADQL&MAXREC={0}&QUERY={1}"
    SEND = TAP_URL.format(MAXREC, qstr)
    print(SEND)

    im = pyfits.open(SEND)
    hi = utils.GTable()
    for c in im[1].data.columns:
        if len(im[1].data[c.name]) > 0:
            hi[c.name] = im[1].data[c.name]
    
    #so = np.argsort(hi['ob_id']*100+hi['tpl_expno'])
    so = np.argsort(hi['exp_start'])
    hi = hi[so]
    
    
def query_eso_archive(filters=['Ks'], rd=(325.6038899,-44.3295833), min_nexp=5, eso=None):
    """
    Query the ESO archive
    """
    from grizli import utils
    from golfir.vlt.archive import get_eso
    
    if eso is None:
        eso = get_eso()
        
    kwargs = {}
    kwargs['column_filters'] = {}
    kwargs['column_filters']['tpl_nexp'] = [f'> {min_nexp}']
    kwargs['column_filters']['tpl_expno'] = [1]
    kwargs['column_filters']['ins_filt1_name'] = filters
    kwargs['column_filters']['dp_cat'] = ['SCIENCE']
    #kwargs['column_filters']['dp_tech'] = ['IMAGE']
    kwargs['column_filters']['pi_coi_name'] = 'PI_only'
    
    #kwargs['column_filters']['RA'] = [f'> {rd[0]-0.5}', f'< {rd[0]}+=0.5']
    
    kwargs['columns'] = ['tpl_nexp', 'tpl_expno', 'det_ndit', 'det_dit', 'det_ncorrs_name', 'obs_tmplno', 'det_ncorrs_name', 'prog_type', 'pi_coi']
    
    #res = eso.query_instrument('hawki', pi_coi_name='PI_only', **kwargs)
    #res['PI'] = [p.split('/')[0].strip() for p in res['PI/CoI']]
    
    kws = {}
    for k in kwargs:
        kws[k] = kwargs[k].copy()
    
    kws['column_filters'].pop('tpl_nexp')
    kws['column_filters'].pop('tpl_expno')
    
    _res = eso.query_instrument('hawki', pi_coi_name='PI_only',
                                coord1=rd[0],
                                coord2=rd[1], 
                                box='00 30 00', 
                                **kws)
    
    field = utils.radec_to_targname(*rd)
                             
    if len(_res) > 0:
        print(f'{field}: {len(_res)} datasets')
        _res['PI'] = [p.split('/')[0].strip() for p in _res['PI/CoI']]
        _res.write(f'{field}_hawki.fits', overwrite=True)
    
    return field, _res
    
    # Run it
    #from golfir.vlt import hawki
    
    runit(root=field, eso=eso, ob_indices=[0], use_hst_radec=False, 
                extensions=[1], redrizzle_args={'use_hst_ref':False,'pad':60}, 
                sync=False, radec=None)
                
    #return eso, kwargs, res
    
def runit(root='j003548m4312', eso=None, ob_indices=None, use_hst_radec=False, extensions=[1,2,3,4], redrizzle_args={'use_hst_ref':True,'pad':60}, fetch=True, request_id=None, clean=False, sync=True, radec=None, **kwargs):
    
    #from golfir.vlt import hawki
    
    # if fetch:
    #     if eso is None:
    #         eso = astroquery.eso.Eso()
    #         eso.login()
    # 
    #     if not eso._authenticated:
    #         eso.login()
        
    if root.startswith('hff-'):
        # Run in two steps for global masking
        pass
    else:
        pass
    
    if __name__ == '__main__':
        ob_indices=None
        use_hst_radec=False
        extensions = [1,2,3,4]
        redrizzle_args = {'pad':60}
        request_id = None
        fetch=True
    
    pipeline(root=root, eso=eso, ob_indices=ob_indices, use_hst_radec=use_hst_radec, radec=radec, extensions=extensions, redrizzle_args=redrizzle_args, fetch=fetch, request_id=request_id, **kwargs)
        
    if sync:
        sync_results(include_exposures=False)
        status = os.path.exists(f'{root}-ks_drz_sci.fits.gz')
    else:
        status = os.path.exists(f'{root}-ks_drz_sci.fits')
        
    os.chdir('../')
    
    if status:
        if clean:
            os.system(f'rm -rf {root}')
        return True
    else:
        print(f'ks mosaic not found.  Problem with {root}?')
        return False
        
def pipeline(root='j234456m6406', eso=None, ob_indices=None, use_hst_radec=False, radec=None, assume_close=20, extensions=[1,2,3,4], fetch=True, request_id=None, redrizzle_args={'use_hst_ref':True, 'pad':60}, ob_minexp=8, retrieve_kwargs=dict(continuation=False, unzip=False), retry_failed=False, **kwargs):
    #from golfir.vlt import hawki

    # if fetch:
    #     if eso is None:
    #         eso = astroquery.eso.Eso()
    #         # Configure 'username' in ~/.astropy/config
    #         eso.login() #reenter_password=True)
    
    if not os.path.exists(f'{root}_hawki.fits .'):
        os.system(f'aws s3 cp s3://grizli-hawki/HAWKI/{root}_hawki.fits .')
    
    tab = utils.read_catalog(root+'_hawki.fits')
    
    if 'TPL EXPNO' in tab.colnames:
        ob_start = np.where(tab['TPL EXPNO'] == 1)[0]
    else:
        ob_start = np.where(tab['tpl_expno'] == 1)[0]
        
    ob_end = np.roll(ob_start, -1)
    ob_end[-1] = len(tab)
    ob_nexp = ob_end - ob_start

    if ob_indices is None:
        ob_indices = np.where(ob_nexp >= ob_minexp)[0]

    datasets = []
    for ind in ob_indices:
        if ind > (len(ob_start)-1):
            print(f'Skip ob_indices={ind} (N={len(ob_start)})')
            continue
            
        sl = slice(ob_start[ind], ob_end[ind])
        if 'access_url' in tab.colnames:
            datasets.extend(list(tab['access_url'][sl]))
        else:
            datasets.extend(list(tab['DP.ID'][sl]))
            
    dirs = [root, os.path.join(root, 'RAW'), os.path.join(root, 'Processed')]
    for dir in dirs:
        print(dir)
        if not os.path.exists(dir):
            os.mkdir(dir)
    
    print(f'{root}: fetch {len(datasets)} files (ob_indices={ob_indices})')       
    
    #request_id = None # Restart an earlier request
        
    os.chdir(dirs[1])

    if fetch & (len(datasets) > 0):
        ND = len(datasets)
        
        if datasets[0].startswith('http'):
            for i, f in enumerate(datasets):
                local_file = os.path.basename(f)+'.fits'
                if os.path.exists(local_file):
                    print(i, local_file, 'Exists')
                else:
                    print(f'\n\n{i+1} / {ND} : Fetch {local_file}\n')
                    os.system(f'wget "{f}" -O {local_file}.Z -nv')
                    os.system(f'gunzip {local_file}.Z')
        else:
            data_files = eso.retrieve_data(datasets, destination=os.getcwd(), request_id=request_id, **retrieve_kwargs)
        
            files = glob.glob('*.Z')
            files.sort()
    
            for file in files: 
                print(file)
                os.system('gunzip '+file)
    
    os.chdir('../')
    
    # os.system(f'aws s3 cp s3://grizli-hawki/Pipeline/{root}/Prep/{root}-ir_drz_sci.fits.gz .')  
        
    # if use_hst_radec:
    #     radec = hawki.make_hst_radec(root, maglim=[16,22])
        
    parse_and_run(extensions=extensions, radec=radec, assume_close=assume_close, max_shift=500, max_rot=3, max_scale=0.02, ob_minexp=ob_minexp, **kwargs)
    
    # Try to rerun the failed observations with the HST reference
    failed = glob.glob('*failed')
    if (len(failed) > 0) & retry_failed:
        for fail in failed:
            froot = fail.split('.failed')[0]
            print(f'Clean {froot} failed')
            os.system(f'rm *{froot}* Processed/{froot}*')
        
        radec = make_hst_radec(root, maglim=[16,22])
        parse_and_run(extensions=[1,2,3,4], radec=radec)
    
    # from importlib import reload
    # reload(hawki)     
    if redrizzle_args is not None:
        redrizzle_mosaics(**redrizzle_args)
    
        
def make_hst_radec(root, maglim=[16,22]):
    
    from grizli import prep
    
    phot_file = f'{root}_phot.fits'
    
    if not os.path.exists(phot_file):
        os.system(f'aws s3 cp s3://grizli-hawki/Pipeline/{root}/Prep/{root}_phot.fits .')  
    
    phot = utils.read_catalog(phot_file)
    sel = (phot['mag_auto'] > maglim[0]) & (phot['mag_auto'] < maglim[1])
    
    radec_file = f'{root}_{maglim[0]:4.1f}_{maglim[1]:4.1f}.radec'
    
    print(f'make_hst_radec: {radec_file} {sel.sum()}')
    
    prep.table_to_radec(phot[sel], radec_file)
    prep.table_to_regions(phot[sel], radec_file.replace('.radec', '.reg'))
    
    return radec_file
    
    
def parse_and_run(extensions=[2], SKIP=True, stop=None, radec=None, ob_minexp=18, seg_file='None', assume_close=10, max_shift=100, max_rot=3, max_scale=0.02, **kwargs):
    
    #from golfir.vlt import hawki
    
    if not os.path.exists('libralato_hawki_chip1.crpix.header'):
        path = os.path.join(os.path.dirname(__file__), '../data')
        os.system(f'cp {path}/libra* .')
    
    os.system('aws s3 sync s3://grizli-stsci/HAWKI/Calibs/ RAW/')
    
    if not os.path.exists('files.list'):        
        os.system('dfits RAW/HAWKI*fits | fitsort OBS.ID DATE-OBS TPL.NEXP TPL.EXPNO | sed "s/FILE/# FILE/" > files.list')
    
    info = utils.read_catalog('files.list')
    first = np.where(info['TPL.EXPNO'] == 1)[0]
    N = len(info)
    
    Ngroups = len(first)
    file_groups = []
    ob_ids = []
    
    for j in range(Ngroups):
        i0 = first[j]
        if j == Ngroups-1:
            i1 = N+1
        else:
            i1 = first[j+1]
        
        Ni = i1-i0
        if Ni > ob_minexp:
            print(info['OBS.ID'][i0], info['DATE-OBS'][i0], i0, i1)
            file_groups.append(info['FILE'][i0:i1].tolist())
            ob_ids.append(info['OBS.ID'][i0])
        else:
            continue
                
    try:
        _x = extensions
    except:
        # Run it
        extensions = [2]
        SKIP = True
        stop = None
        radec = None
        
    # Groups
    group_dict = {}
    for i, sci_files in enumerate(file_groups):
        for ext in extensions:
            ob_root = get_ob_root(sci_files[0])            
            key = '{0}-{1}'.format(ob_root, ext)
            group_dict[key] = sci_files
    
    Ng = len(group_dict)    
    for i, key in enumerate(group_dict):
        
        sci_files = group_dict[key]
        ext = int(key[-1])
        ob_root = key[:-2]
        
        drz_file = '{0}-{1}-ks_drz_sci.fits'.format(ob_root, ext)
        if os.path.exists(drz_file) & SKIP:
            print('Skip ', i, Ng, ob_root, ext, len(sci_files))
            continue
        else:
            print(i, Ng, ob_root, ext, len(sci_files))
        
        # if ext in [1,4]:
        #     radec = 'hff-j001408m3023_master.radec'
        # else:
        #     radec = None
        if 'Process2020' in os.getcwd():
            radec = 'a2744_hst+gaia.radec' # for first few
            radec = 'a2744_ks_shallow.radec'
            seg_file = 'xa2744-test-ks_seg.fits'
            
            #radec = 'a2744-remask_wcs.radec'
            seg_file = 'a2744-remask_seg.fits'
            
        elif 'alentino' in os.getcwd():
            radec = 'hsc_21.radec'
            seg_file = 'None'
        else:
            #radec = None # GAIA DR2
            pass
            #seg_file = 'None'
            
        try:
            process_hawki(sci_files, bkg_order=3, ext=ext, ds9=None, bkg_percentile=47.5, assume_close=assume_close, radec=radec, stop=stop, seg_file=seg_file, max_shift=max_shift, max_rot=max_rot, max_scale=max_scale, **kwargs)

            LOGFILE = '{0}-{1}.failed'.format(ob_root, ext)
            if os.path.exists(LOGFILE):
                os.remove(LOGFILE)
        except:
            LOGFILE = '{0}-{1}.failed'.format(ob_root, ext)
            utils.log_exception(LOGFILE, traceback)
    
def sync_results(include_exposures=False):
        
    # Zip and Sync to AWS
    root = os.getcwd().split('/')[-1]

    # Remove old Failed files
    os.system(f'aws s3 rm s3://grizli-hawki/HAWKI/{root}/ --recursive --exclude "*" --include "*failed"')

    files = glob.glob(f'{root}-ks_drz*')    
    files += glob.glob('*flat.masked.fits')
    files += glob.glob('ob*_drz*fits')
    if include_exposures:
        files += glob.glob('Processed/*sci.fits')
    
    for file in files:
        print(f'gzip {file}')
        os.system(f'gzip {file}')
        
    fp = open(f'{root}_hawki.html','w')
    fp.write(f'<h4>{root} - {time.ctime()}</h4>\n')
    fp.write(f'HST: <a href="https://s3.amazonaws.com/grizli-hawki/Pipeline/{root}/Prep/{root}.summary.html">{root}</a><br>\n')

    fp.write(f'<img src="https://s3.amazonaws.com/grizli-hawki/HAWKI/{root}_hawki.png" height=300px><br>\n')
    fp.write(f'<a href="https://s3.amazonaws.com/grizli-hawki/HAWKI/{root}_hawki.fits">{root}_hawki.fits</a><br><p>\n')
    
    fp.write('<pre>\n')
    
    cmd = f'aws s3 sync ./ s3://grizli-hawki/HAWKI/{root}/ --acl public-read --exclude "*" --include "*html"'
    
    os.system(f'dfits Processed/*fits | fitsort  CRVAL1 CRVAL2 CRPIX1 CRPIX2 CD1_1 CD1_2 CD2_1 CD2_2 BKGORDER BKG001 BKG002 BKG003 BKG003 BKG004 BKG005 BKG006 BKG007 BKG008 BKG009 ORIGFILE DET.NDIT DET.DIT TPL.NEXP TPL.EXPNO OBS.ID > {root}.exposures.txt')
    
    for group in [f'{root}-ks_drz*gz', '*flat.masked.fits.gz', '*files',  'ob*_drz*fits.gz', 'ob*[1-4].log', '*exposures.txt', 'ob*failed', 'ob*[1-4]*wcs.*', 'Processed/*gz', 'vista.fits', '*gaia*']:
        files = glob.glob(group)
        files.sort()
        
        if len(files) == 0:
            continue
            
        for file in files:
            fp.write(f'<a href="{file}">{file}</a>\n')
        
        fp.write('\n')
        cmd += f' --include "{group}"'
    
    fp.write('</pre>')
    fp.close()
    os.system(cmd) 
    
def flat_gradient():
    """
    Average gradient in the chip flats
    """
    for ext in [1,2,3,4]:
        flat_files = glob.glob('*-{0}.flat.fits'.format(ext))
        flat_files.sort()
        print('flat gradient', ext, len(flat_files))
        flats = [pyfits.open(file)['POLY'].data for file in flat_files]
        f = np.mean(np.array(flats), axis=0)
        pyfits.writeto('flat_gradient-{0}.fits'.format(ext), data=f, overwrite=True)
        
def redrizzle_mosaics(cat_kwargs={}, use_hst_ref=True, pad=60):
    """
    Redrizzle mosaic with padding
    
    pad: padding around ref_image in arcsec
    
    """
    if 'Process2020' in os.getcwd():
        ref_image = 'hff-j001408m3023-f160w_drz_sci.fits.gz'
        root = 'a2744-test'
    elif 'alentino' in os.getcwd():
        ref_image = 'hawki-driz-v2-100mas-ks_drz_sci.fits'
        root = 'valentino-qz4'
    else:
        field = os.getcwd().split('/')[-1]
        ref_image = f'{field}-ir_drz_sci.fits.gz'
        root = f'{field}-ks'
        
    if os.path.exists(ref_image) & use_hst_ref:
        ref = pyfits.open(ref_image)
    else:
        files = glob.glob('Processed/*sci.fits')
        valid_wcs = []
        for file in files:
            im = pyfits.open(file)
            valid_wcs.append(pywcs.WCS(im[0].header))
        
        ref_hdu = utils.make_maximal_wcs(valid_wcs, pixel_scale=0.1, pad=0)
        ref = [ref_hdu]
        
    sh = ref[0].data.shape
    ref_header = ref[0].header.copy()
    
    if pad > 0:
        ref_wcs = pywcs.WCS(ref_header, relax=True)
        ref_wcs.pscale = utils.get_wcs_pscale(ref_wcs)

        pad_pix = int(np.round(pad/ref_wcs.pscale))
        ref_header['NAXIS1'] += 2*pad_pix
        ref_header['NAXIS2'] += 2*pad_pix
        ref_header['CRPIX1'] += pad_pix
        ref_header['CRPIX2'] += pad_pix

    ref_wcs = pywcs.WCS(ref_header, relax=True)
    ref_wcs.pscale = utils.get_wcs_pscale(ref_wcs)
    
    if False:
        ref_header, ref_wcs = utils.make_maximal_wcs([ref_wcs], pixel_scale=0.1, theta=0, pad=60, get_hdu=False, verbose=True)
    
    xfiles = glob.glob('*.flat.masked.fits')
    xfiles.sort()
    
    for xfile in xfiles:
        froot = os.path.basename(xfile).split('.flat.masked.fits')[0]
        ext_i = int(froot[-1])
        ob_root = froot[:-2]
        
        out_root = f'mosaic-{ob_root}-{ext_i}-ks'
        files = glob.glob(f'Processed/{ob_root}*-{ext_i}.sci.fits')
        N = len(files)
        
        if os.path.exists(f'{ob_root}-{ext_i}.failed'):
            print('Failed', ob_root, ext_i)
            
        if os.path.exists(f'{out_root}_drz_sci.fits') | (N == 0):
            print('Skip', out_root, N)
            continue
        else:
            print(out_root, N)
        
        dq = pyfits.open(f'{ob_root}-{ext_i}.flat.fits')
        
        _drz = None
        for i, file in enumerate(files):
            print(i, N, file)
            im = pyfits.open(file)
            wcs_i = pywcs.WCS(im[0].header, relax=True)
            wcs_i.pscale = utils.get_wcs_pscale(wcs_i)
            
            sci = im[0].data
            wht = dq['DQ'].data/im[0].header['NMAD']**2

            if _drz is None:
                _drz = utils.drizzle_array_groups([sci], [wht.astype(np.float32)], [wcs_i], outputwcs=ref_wcs, kernel='point', verbose=False)
                header = _drz[3]
                header['NDRIZIM'] = 1
                header['DIT'] = 15.
                #header['EXPTIME'] = im[0].header['EXPTIME']
            else:
                _ = utils.drizzle_array_groups([sci], [wht.astype(np.float32)], [wcs_i], outputwcs=ref_wcs, kernel='point', data=_drz[:3], verbose=False)
                header['NDRIZIM'] += 1
                #header['EXPTIME'] += im[0].header['EXPTIME']    

        relative_pscale = ref_wcs.pscale/wcs_i.pscale
        sci_scl = 1./header['DIT'] 
        wht_scl = header['DIT']**2*relative_pscale**-4

        header['ZP'] = 25, 'Dummy zeropoint'
        header['PHOTFNU'] = 10**(-0.4*(25-8.90)), 'Dummy zeropoint'
        header['FILTER'] = 'Ks'
        header['INSTRUME'] = 'HAWKI'

        print('Write')
        pyfits.writeto('{0}_drz_sci.fits'.format(out_root), data=_drz[0]*sci_scl, header=header, clobber=True, output_verify='fix')
        pyfits.writeto('{0}_drz_wht.fits'.format(out_root), data=_drz[1]*wht_scl, header=header, clobber=True, output_verify='fix')
        
        if _drz[0].max() == 0:
            print('Empty mosaic: {0}'.format(out_root))
            continue
            
        bkg_params={'bw': 128, 'bh': 128, 'fw': 3, 'fh': 3, 'pixel_scale':0.1}

        cat = prep.make_SEP_catalog(out_root, threshold=1.2, column_case=str.lower, bkg_params=bkg_params, **cat_kwargs)
        
        # Masked background
        seg = pyfits.open('{0}_seg.fits'.format(out_root))
        seg_mask = seg[0].data > 0

        cat = prep.make_SEP_catalog(out_root, threshold=1.4, column_case=str.lower, bkg_params=bkg_params, bkg_mask=seg_mask, **cat_kwargs)
        
    # Combined    
    # query vizier for VISTA surveys
    ra, dec = ref_wcs.wcs.crval

    vista = prep.query_tap_catalog(db='"II/343/viking2"', ra=ra, dec=dec, radius=12, vizier=True, extra='AND Kspmag > 10')
    if len(vista) < 10:
        vista2 = prep.query_tap_catalog(db='"II/359/vhs_dr4"', ra=ra, dec=dec, radius=12, vizier=True, extra='AND Kspmag > 10')
        if (len(vista2) > 0) & (len(vista2) > len(vista)):
            vista = vista2
    
    vega2ab = 1.827
    kcol = 'Kspmag'
    ekcol = 'e_'+kcol
    
    if len(vista) == 0:
        twom = prep.query_tap_catalog(db='"II/246/out"', ra=ra, dec=dec, radius=12, vizier=True, extra='AND e_Kmag > 0 AND e_Kmag < 0.3')
        if (len(twom) > 0) & (len(twom) > len(vista)):
            vista = twom
            kcol = 'Kmag'
            ekcol = 'e_Kmag'
                
    vista = utils.GTable(vista)
    vista['ra'] = vista['RAJ2000']
    vista['dec'] = vista['DEJ2000']

    vista.write('vista.ecsv', overwrite=True)
    #################
    
    vista = utils.read_catalog('vista.ecsv', format='ascii.ecsv')
    
    vista['ra'] = vista['RAJ2000']
    vista['dec'] = vista['DEJ2000']
    prep.table_to_regions(vista, 'vista.reg')
    
    if False:
        # Use Full K as reference
        vista = utils.read_catalog('a2744-remask.cat.fits')
        vega2ab = 0
        kcol = 'mag_auto'
        ekcol = 'magerr_auto'
    
    stars = None
    
    num = None
    
    fig = plt.figure(figsize=[10,6])
    a1 = fig.add_subplot(121)
    a2 = fig.add_subplot(122)
    
    for ix, xfile in enumerate(xfiles):
        froot = os.path.basename(xfile).split('.flat.masked.fits')[0]
        ext_i = int(froot[-1])
        ob_root = froot[:-2]
        
        out_root = f'mosaic-{ob_root}-{ext_i}-ks'
        cat_file = f'{out_root}.cat.fits'
        if not os.path.exists(cat_file):
            continue
            
        cat = utils.read_catalog(cat_file)

        #plt.scatter(cat['mag_auto'.lower()], cat['flux_radius'.lower()], alpha=0.2)
            
        if len(vista) > 0:
            idx, dr = vista.match_to_catalog_sky(cat)
        
            try:
                vkmag = vista[idx][kcol].filled()
                vkmagerr = vista[idx][ekcol].filled()
            except:
                vkmag = vista[idx][kcol]
                vkmagerr = vista[idx][ekcol]
            
            mat = (dr.value < 0.5) & (vkmag > 12) & (vkmag < 22)
        
            mat &= np.isfinite(cat['mag_auto'])
        
            # http://casu.ast.cam.ac.uk/surveys-projects/vista/technical/filter-set
        
            dmag = cat['mag_auto'] - (vkmag + vega2ab)
            mat &= np.isfinite(dmag) & (cat['flux_radius'] > 1) & (cat['mag_auto'] < 20) #& (vista[ekcol][idx].filled() < 0.2)
        
            try:
                mat &= ~dmag.mask
            except:
                pass
        
            if mat.sum() > 64:
                limits = np.percentile(cat['mag_auto'][mat], [5, 20])
                mat &= (cat['mag_auto'] > limits[0]) & (cat['mag_auto'] < limits[1])
            
            med_dmag = np.median(dmag[mat])
            mat &= np.abs(dmag - med_dmag) < 0.5
        
        else:
            mat = np.zeros(len(cat), dtype=bool)
            
        if mat.sum() > 2:
            med_dmag = np.median(dmag[mat])
            is_calibrated = True
            
            a1.errorbar(cat['mag_auto'][mat] - med_dmag, dmag[mat], vista[ekcol][idx][mat],  alpha=0.1, linestyle='None', marker='None', color='k')
            a1.scatter(cat['mag_auto'][mat] - med_dmag, dmag[mat], zorder=1000, alpha=0.5)
            a1.set_ylim(med_dmag-1, med_dmag+1)
                        
        else:
            med_dmag = 0 
            is_calibrated = False
            ref_mag = {1:22.398, 2:22.633, 3:22.6609, 4:22.409}
            med_dmag = ref_mag[ext_i] - 25
        
        sci = pyfits.open(f'{out_root}_drz_sci.fits')
        wht = pyfits.open(f'{out_root}_drz_wht.fits')[0].data
            
        phot_scl = 10**(0.4*med_dmag)
        
        ZPi = 25+med_dmag        
        #ZP = 26
        ZP = 23.9
        zp26 = 10**(-0.4*(25-ZP)) # 26
        
        # Image quality
        if stars is not None:
            sidx, sdr = stars.match_to_catalog_sky(cat)
            smat = sdr.value < 0.1
            flux_ratio = (cat['flux_aper_0']/cat['flux_aper_6'])
            smat *= np.isfinite(flux_ratio) & (flux_ratio > 0) & (flux_ratio < 1)
            a2.scatter(cat['mag_auto'][smat] - med_dmag, flux_ratio[smat])
            med_ratio = np.median(flux_ratio[smat])
            ellip = 1 - cat['b_image']/cat['a_image'] 
            med_ellip = np.median(ellip[smat])
            
            if smat.sum() == 0:
                med_ratio = -1
                med_ellip = -1
                
        else:
            med_ratio = -1
            med_ellip = -1

        ### Weight scaling for transparency
        wht_scl = (0.1/phot_scl)**2

        ### Weight scaling for image quality        
        if med_ratio > 0:
            wht_scl *= (med_ratio/0.6)**2
        
        ### Weight scaling for ellipticity
        if med_ellip > 0:
            wht_scl *= np.sqrt(1-med_ellip**2)
            
        print(ix, '{0} scl={1:6.3f}, AB={2:6.2f} ({3}) flux_ratio  = {4:.2f}, ellip = {5:.2f}, WHT = {6:.2f}'.format(xfile.split('.flat')[0], phot_scl, ZPi, mat.sum(), med_ratio, med_ellip, wht_scl))

        xsci = pyfits.open(f'{out_root}_drz_sci.fits', mode='update')
        xsci[0].header['CALZP'] = ZPi, 'Calibrated AB ZP from VISTA'
        xsci[0].header['PHOT_SCL'] = phot_scl, 'Flux scaling to AB'
        xsci[0].header['NCAL'] = mat.sum(), 'Number of sources for calibration'
        xsci[0].header['CALDB'] = vista.meta['TAPDB'][0], 'Table of reference sources'
        xsci[0].header['IS_CALIB'] = is_calibrated, 'Magnitudes are calibrated from Vista'    
        xsci[0].header['FRATIO'] = med_ratio, 'Aperture ratio as image quality'    
        xsci[0].header['ELLIP'] = med_ellip, 'Stellar ellipticity'    
        
        xsci[0].header['WHTSCL'] = wht_scl, 'Scale factor applied to weights'
            
        xsci.flush()
                
        # Apply to images
        sci[0].data *= phot_scl*zp26
        wht *= 1./(phot_scl*zp26)**2*wht_scl
        
        if num is None:
            num = sci[0].data*wht
            den = wht
            header = sci[0].header.copy()
        else:
            num += sci[0].data*wht
            den += wht
            header['NDRIZIM'] += sci[0].header['NDRIZIM']
    
    a1.set_xlabel('mag_auto')
    a1.set_xlim(15, 20)
    a2.set_ylabel('delta mag')
    a2.set_ylim(0, 1)
    a2.set_xlim(a1.get_xlim())
    a2.set_xlabel('mag_auto')
    a1.grid(); a2.grid()
    fig.tight_layout(pad=0.2)
    
    fig.savefig(root+'_phot.png')
    
    sci = num/den
    sci[den == 0] = 0
    
    header['ZP'] = ZP, 'AB zeropoint'
    header['PHOTFNU'] = 10**(-0.4*(ZP-8.90)), 'AB zeropoint'
    header['FILTER'] = 'Ks'
    header['INSTRUME'] = 'HAWKI'
    
    # Full catalog
    pyfits.writeto(root+'_drz_sci.fits', data=sci, header=header, overwrite=True)
    pyfits.writeto(root+'_drz_wht.fits', data=den, header=header, overwrite=True)
    
    if False:
        bkg_params={'bw': 128, 'bh': 128, 'fw': 3, 'fh': 3, 'pixel_scale':0.1}
    
        cat = prep.make_SEP_catalog(root, threshold=1.2, column_case=str.lower, bkg_params=bkg_params)
    
        # Masked background
        seg = pyfits.open('{0}_seg.fits'.format(root))
        seg_mask = seg[0].data > 0
    
        cat = prep.make_SEP_catalog(root, threshold=1.2, column_case=str.lower, bkg_params=bkg_params, bkg_mask=seg_mask)
     
        # Point sources
        stars = (cat['mag_auto'] < 21.5) & (cat['mag_auto'] > 16) & (cat['flux_radius'] < 3.2) & (cat['flux_radius'] > 2)
        cat['ra', 'dec', 'mag_auto'][stars].write('a2744_stars.fits')
        
def get_ob_root(h0):
    """
    Get the OB root name from the header
    """
    if isinstance(h0, str):
        h0 = pyfits.open(h0)[0].header
    
    ob_root = 'ob{0}-{1}'.format(h0['HIERARCH ESO OBS ID'], h0['DATE-OBS'][:-5].replace('-','').replace(':','').replace('T','-'))
    
    return ob_root


def process_hawki(sci_files, bkg_order=3, ext=1, ds9=None, bkg_percentile=50, assume_close=10, radec=None, stop=None, seg_file='None', max_shift=100, max_rot=3, max_scale=0.02, n_brightest=100, align_mag_limits=[6,28,0.15], align_clip=500, **kwargs):
    """
    Reduce a HAWKI chip
    """
    from scipy.spatial import cKDTree
    from skimage.transform import SimilarityTransform, EuclideanTransform
    
    # Default calibrations
    flat = pyfits.open('RAW/M.HAWKI.2019-10-22T08_04_04.870.fits')
    gain = pyfits.open('RAW/M.HAWKI.2019-10-22T08_04_11.606.fits')
    dark = pyfits.open('RAW/M.HAWKI.2019-10-24T07_34_33.323.fits')
    
    ob_root = get_ob_root(sci_files[0])
    
    fp = open(ob_root+'.files','w')
    for file in sci_files:
        fp.write(file+'\n')
    fp.close()
    
    NFILES = len(sci_files)
    LOGFILE = '{0}-{1}.log'.format(ob_root, ext)
    
    skip_list = np.zeros(NFILES, dtype=bool)
    
    primary_headers = []
    for i in range(NFILES):
        head_i = pyfits.open(sci_files[i])[0].header.copy()
        primary_headers.append(head_i)
    
    sci = pyfits.open(sci_files[0])
    h0 = sci[0].header
    
    msg = f"""
###################
#
# ob_root = {ob_root}
#     ext = {ext}
#       N = {NFILES}
#    """

    utils.log_comment(None, msg, verbose=True, show_date=True)

    #################
    msg = '{0}-{1}: Read data, {2} files'.format(ob_root, ext, NFILES)
    utils.log_comment(LOGFILE, msg, verbose=True, show_date=True, mode='a')
    
    sci1 = np.array([pyfits.open(file)[ext].data.flatten() for file in sci_files])
    
    if dark is not None:
        dkey = 'HIERARCH ESO DET DIT'
        dscale = h0[dkey]/dark[0].header[dkey]
        sci1 -= dark[ext].data.flatten()*dscale
    
    sh = (2048,2048)
    py, px = np.indices(sh)
    px = px.flatten()
    py = py.flatten()
    
    pyn, pxn = (np.indices(sh)-1024)/1024
    
    ######### Background model
    if True:
        # Hermite polynomials for background
        _A = []
        horder = bkg_order
        from numpy.polynomial.hermite import hermgrid2d
        xa = np.linspace(-1,1,2048)
        ct = 0
        for i in range(horder):
            for j in range(horder):
                c = np.zeros((horder,horder))
                c[i,j] = 1
                herm = hermgrid2d(xa, xa, c)
                #ds9.frame(7+ct)
                #ds9.view(herm)
                ct+=1
                _A.append(herm.flatten())
    
        _A = np.array(_A)
    else:
        _A = np.array([py.flatten()*0+1, pxn.flatten(), pyn.flatten()])
                
    dq = (px > 100) & (py > 100) & (px < 2048-100) & (py < 2048-100)
        
    ###### Flat
    flat_files = glob.glob('{0}-{1}.flat*fits'.format(ob_root, ext))
    flat_files.sort()
    
    pad = 100
    
    if len(flat_files) == 0:
                
        ################
        bg0 = np.median(sci1[:,dq], axis=1)
    
        if bkg_percentile == 50:
            # Median
            print('Get median for flat')
            med = np.median(sci1.T/bg0, axis=1)
        else:
            # Reject top 5%
            print('Get percentile for flat')
            level = int(np.round(bkg_percentile/100*NFILES))/NFILES*100
            med = np.percentile(sci1.T/bg0, level, axis=1)
    
        med0 = np.median(med) 
        flat1 = med/med0
    
        dq = np.isfinite(flat1) & (flat1 > 0.2) & (flat1 < 5) & (px > pad) & (py > pad) & (px < 2048-pad) & (py < 2048-pad)
    
        if dark is not None:
            dq *= ((dark[ext].data > -100) & (dark[ext].data < 210)).flatten()
    
        # "flatten" the flat
        _fres = np.linalg.lstsq(_A[:,dq].T, flat1[dq], rcond='warn')
        flat_sm = _A.T.dot(_fres[0])
        flat1 /= flat_sm
        flat1[flat1 == 0] = 1
    
        dq &= (flat1 > 0)
    
        h = pyfits.Header()
        h['NFILES'] = NFILES
        for i, f in enumerate(sci_files):
            h['FILE{0:04d}'.format(i)] = sci_files[i]
        
        hdu = pyfits.HDUList(pyfits.PrimaryHDU(data=flat1.reshape(sh), header=h))
    
        h = pyfits.Header()
        h['BKGORDER'] = bkg_order
        for j in range(len(_fres[0])):
            h['BKG{0:03d}'.format(j)] = _fres[0][j]
    
        hdu.append(pyfits.ImageHDU(data=flat_sm.reshape(sh), header=h, name='POLY'))
        hdu.append(pyfits.ImageHDU(data=dq.reshape(sh)*1, name='DQ'))
        
        flat_file = f'{ob_root}-{ext}.flat.fits'
        hdu.writeto(flat_file, overwrite=True)
    
        get_masked_flat = True
    
    else:
        flat_file = flat_files[-1]
        
        get_masked_flat = 'masked' not in flat_file
        
        print('Read flat file ', flat_file)
        flat1 = pyfits.open(flat_file)[0].data.flatten()
        
        dq = np.isfinite(flat1) & (flat1 > 0.2) & (flat1 < 5) 
        dq &= (px > pad) & (py > pad) & (px < 2048-pad) & (py < 2048-pad)

        if dark is not None:
            dq *= ((dark[ext].data > -100) & (dark[ext].data < 210)).flatten()
    
    # (bug, gradient already in flat2)
    if 'masked' not in flat_file:
        gradient_file = 'flat_gradient-{0}.fits'.format(ext)
        if os.path.exists(gradient_file):
            print('Use flat gradient file ', gradient_file)
            grad = pyfits.open(gradient_file)
            flat1 *= grad[0].data.flatten()
    
    if stop == 'flat':
        return True
    
    ############# WCS
    # SIP header from Libralato
    print('Read WCS')
    if 1:
        sip = pyfits.Header.fromtextfile('libralato_hawki_chip{0}.crpix.header'.format(ext))
    elif 0:
        sip = pyfits.Header.fromfile('libralato_hawki_chip{0}.header'.format(ext))
    else: 
        sip = {}
    
    # Native WCS
    wcs_list = []
    headers = []
    for i in range(NFILES):
        head_i = pyfits.open(sci_files[i])[ext].header.copy()
        for k in sip:
            if k in ['COMMENT']:
                continue

            head_i[k] = sip[k]
        
        headers.append(head_i)
        wcs_i = pywcs.WCS(head_i, relax=True)
        wcs_i.pscale = utils.get_wcs_pscale(wcs_i)
        wcs_list.append(wcs_i)
        
    # Initial background subtraction
    pad=50
    dq = np.isfinite(flat1) & (flat1 > 0.2) & (flat1 < 5) & (px > pad) & (py > pad) & (px < 2048-pad) & (py < 2048-pad)
    if dark is not None:
        dq *= ((dark[ext].data > -100) & (dark[ext].data < 210)).flatten()
    
    dqsh = dq.reshape(sh)
    sci_list = []
    wht_list = []

    dq &= (flat1 != 0)
    
    for i in range(NFILES):
        sci_i = sci1[i,:]/flat1
        sci_i[~dq] = 0
        
        _bres = np.linalg.lstsq(_A[:,dq].T, sci_i[dq], rcond=None)
        bg_i = _A.T.dot(_bres[0])

        msg = 'Initial background ({2:>2}): {0}[{1}] bg0={3:7.1f}'.format(sci_files[i], ext, i, _bres[0][0])
        utils.log_comment(LOGFILE, msg, verbose=True)

        if ds9 is not None:
            ds9.view(((sci_i - bg_i)*dq).reshape(sh)/1000, header=utils.to_header(wcs_list[i]))

        sci_list.append((sci_i - bg_i).reshape(sh))
        wht_i =  np.cast[np.float32](dqsh/utils.nmad(sci_list[i][dqsh])**2)
        wht_list.append(wht_i)
    
    ###################      
    # Alignment
    ####### GAIA DR2 catalog
    if radec is None:
        r0 = h0['RA']
        d0 = h0['DEC']
    
        # GAIA catalog
        try:
            gaia = prep.get_gaia_DR2_catalog(ra=r0, dec=d0, radius=8, use_mirror=False, output_file='{0}-gaia.fits'.format(ob_root)) 
        except:
            gaia = prep.get_gaia_DR2_vizier(ra=r0, dec=d0, radius=8) 
            gaia.write('{0}-gaia.fits'.format(ob_root), overwrite=True)
                        
        t0 = Time(sci[0].header['DATE'].replace('T',' '))
        gaia_pm =  prep.get_gaia_radec_at_time(gaia, date=t0.decimalyear, format='decimalyear')
        ok = np.isfinite(gaia_pm.ra)
        gaia_pm = gaia_pm[ok]
    
        gtab = utils.GTable()
        gtab['ra'] = gaia_pm.ra
        gtab['dec'] = gaia_pm.dec
    
        prep.table_to_radec(gtab, '{0}-gaia.radec'.format(ob_root))
        prep.table_to_regions(gtab, '{0}-gaia.reg'.format(ob_root))
    
        radec = '{0}-gaia.radec'.format(ob_root)
    
    radec_tab = utils.read_catalog(radec)
    
    # Exposure catalogs
    fig = plt.figure()
    ax = plt.gca()
    cat_list = []
    for i in range(NFILES):
        if skip_list[i]:
            continue
        
        err = 1/np.sqrt(wht_list[i])
        mask = ~np.isfinite(err)
        err[mask] = 1e10
        sci_i = sci_list[i].byteswap().newbyteorder()
        
        cat_i, seg = prep.make_SEP_catalog_from_arrays(sci_i, err, mask, wcs=wcs_list[i], threshold=7, verbose=False)
        ok = cat_i['flux'] > 0
        cat_i = cat_i[ok]

        if n_brightest is not None:
            so = np.argsort(cat_i['flux'])[::-1]
            if len(so) > n_brightest+10:
                cat_i = cat_i[so][10:n_brightest+10]
            else:
                cat_i = cat_i[so][:n_brightest]
                
        msg = 'Exposure catalogs ({2}): {0}[{1}], N={3}'.format(sci_files[i], ext, i, len(cat_i))
        utils.log_comment(LOGFILE, msg, verbose=True)
        #print('Exposure catalogs', sci_files[i], ext, i)
                    
        cat_i = cat_i[cat_i['y'] > 100]
        
        if len(cat_i) < 6:
            msg = '   ! Skip {0}[{1}], not enough objects found  N={2}'.format(sci_files[i], ext, len(cat_i))
            utils.log_comment(LOGFILE, msg, verbose=True)
            skip_list[i] = True
            wht_list[i] *= 0.
            
        cat_list.append(cat_i)
        ax.scatter(cat_i['ra'], cat_i['dec'], alpha=0.1, marker='.')
    
    ax.set_title(f'{ob_root}-{ext}')
    fig.savefig(f'{ob_root}-{ext}.expsrc.png')
    ax.grid()
    fig.tight_layout(pad=0.1)
    plt.close(fig)
    
    i0=3
    maxKeep = 4
    auto = 5
    
    use_exposures = True
    ransac = True
    cat = None
    
    ############################
    for align_iter in range(2):
        off = 1024
        xoff = yoff = 0
        if True:
            xoff = sci[ext].header['CRPIX1']
            yoff = sci[ext].header['CRPIX2']
    
        if use_exposures:
            ref_wcs = wcs_list
            transform_wcs = []
        else:
            ref_wcs = transform_wcs
                    
        for i in range(NFILES):
            
            if skip_list[i]:
                if use_exposures:
                    transform_wcs.append(wcs_list[i])
                continue
                
            if use_exposures:
                ix = i0
            else:
                ix = i
            
            if use_exposures:
                align_type = 'exposures'
                V1 = np.vstack([cat_list[i0]['x']-xoff, cat_list[i0]['y']-yoff]).T
            elif 0:
                align_type = 'gaia'
                x_i, y_i = ref_wcs[i].all_world2pix(gaia_pm.ra, gaia_pm.dec, 0)
                V1 = np.vstack([x_i, y_i]).T
                clip = ((V1 > 50) & (V1 < 2018)).sum(axis=1) == 2
                V1 = V1[clip,:]- np.array([xoff, yoff])
        
            else:
                # cat is derived from the global stack in second iteration
                align_type = f'ext ({radec})'
                # ref_ra = ps1['ra']
                # ref_dec = ps1['dec']
            
                try:
                    so = np.argsort(cat['MAG_AUTO'])
                    ref_ra = cat['RA'][so][:40]
                    ref_dec = cat['DEC'][so][:40]
                except:
                    np.random.seed(1)
                    ref_ra = cat['ra']*1
                    ref_dec = cat['dec']*1
                    
                x_i, y_i = ref_wcs[ix].all_world2pix(ref_ra, ref_dec, 0)
                V1 = np.vstack([x_i, y_i]).T
                clip = ((V1 > 50) & (V1 < 2018)).sum(axis=1) == 2
                V1 = V1[clip,:] - np.array([xoff, yoff])
        
            if use_exposures:
                x_i, y_i = ref_wcs[ix].all_world2pix(cat_list[i]['ra'], cat_list[i]['dec'], 0)
            else:
                x_i, y_i = cat_list[i]['x'], cat_list[i]['y']
                
            V2 = np.vstack([x_i, y_i]).T - np.array([xoff, yoff])
            #V2 = np.vstack([cat_list[i]['x'], cat_list[i]['y']]).T-1024
    
            try:
                pair_ix = match.match_catalog_tri(V1, V2, maxKeep=maxKeep, auto_keep=auto, ignore_rot=True, ignore_scale=True, ba_max=0.98, size_limit=[1,2000])#, ignore_rot=True, ignore_scale=True, ba_max=0.98, size_limit=[1,1000])
            except:
                
                msg = '   ! Alignment failed {0}[{1}],   N={2}'.format(sci_files[i], ext, len(cat_list[i]))
                utils.log_comment(LOGFILE, msg, verbose=True)
                
                utils.log_exception(LOGFILE, traceback)
                
                skip_list[i] = True
                wht_list[i] *= 0.
                if use_exposures:
                    transform_wcs.append(wcs_list[i])
                
                continue
                    
            try:
                tfo, dx, rms = match.get_transform(V1, V2, pair_ix, transform=SimilarityTransform, use_ransac=ransac)
                if not hasattr(tfo, 'scale'):
                    tfo.scale = 1.
                    
            except:
                msg = '   ! Align transform failed {0}[{1}],   N={2}'.format(sci_files[i], ext, len(cat_list[i]))
                utils.log_comment(LOGFILE, msg, verbose=True)
                
                utils.log_exception(LOGFILE, traceback)
                
                skip_list[i] = True
                wht_list[i] *= 0.
                if use_exposures:
                    transform_wcs.append(wcs_list[i])
                
                continue
              
            fig = match.match_diagnostic_plot(V1, V2, pair_ix, tf=tfo,
                                              new_figure=True)
            fig.savefig('transform_ext{0}_{1:02d}.png'.format(ext, i))
            #print(align_type, i, len(pair_ix), tfo.translation, tfo.rotation, tfo.scale, rms)            
            plt.close('all')
            
            msg = f'  Align with {align_type} ({i:3}) npairs = {len(pair_ix)}, [{tfo.translation[0]:5.2f}, {tfo.translation[1]:5.2f}] {tfo.rotation/np.pi*180:6.3f} {tfo.scale:5.3f} (rms {rms[0]:4.2f} {rms[1]:4.2f})'
            utils.log_comment(LOGFILE, msg, verbose=True)
            
            if (np.max(np.abs(tfo.translation)) > max_shift) | (np.abs(tfo.rotation/np.pi*180) > max_rot) | (np.abs(np.log10(tfo.scale)) > max_scale): 
                msg = f'  ! {msg} !!! bad transform'
                utils.log_comment(LOGFILE, msg, verbose=True)
                
                skip_list[i] = True
                wht_list[i] *= 0.
                if use_exposures:
                    transform_wcs.append(wcs_list[i])
                
                continue
              
            
            if use_exposures:
                transform_wcs.append(utils.transform_wcs(wcs_list[i], translation=tfo.translation, rotation=tfo.rotation, scale=tfo.scale))
            else:
                upd_wcs = utils.transform_wcs(transform_wcs[i].copy(), translation=tfo.translation, rotation=tfo.rotation, scale=tfo.scale)

                transform_wcs[i] = upd_wcs
            
            transform_wcs[i]._naxis = [2048,2048]
            transform_wcs[i].tf = tfo
            
            # Update catalog coordinates
            cat_list[i]['ra'], cat_list[i]['dec'] = transform_wcs[i].all_pix2world(cat_list[i]['x'], cat_list[i]['y'], 0)
           
        #out_h, out_wcs = utils.make_wcsheader(ra=34.28732461, dec=-5.049517927, size=180, pixscale=0.1)
        
        valid_wcs = []
        for i in range(NFILES):
            if not skip_list[i]:
                valid_wcs.append(transform_wcs[i])
                
        max_h, max_wcs = utils.make_maximal_wcs(valid_wcs, pixel_scale=0.1, get_hdu=False, pad=1, verbose=True)
             
        #_drz = utils.drizzle_array_groups(sci_list, wht_list, wcs_list, outputwcs=max_wcs, kernel='point')
        _drzt = utils.drizzle_array_groups(sci_list, wht_list, transform_wcs, outputwcs=max_wcs, kernel='point')
    
        max_h['EXTNAME'] = headers[0]['EXTNAME']
        max_h['FILTER'] = 'Ks'
        max_h['INSTRUME'] = 'HAWKI'
        max_h['TELESCOP'] = 'ESO-VLT-U4'
        max_h['DIT'] = h0['HIERARCH ESO DET DIT']
        max_h['NDIT'] = h0['HIERARCH ESO DET NDIT'] # TPL.EXPNO
        max_h['NEXP'] = h0['HIERARCH ESO TPL NEXP']
        max_h['OBSID'] = h0['HIERARCH ESO OBS ID']
        max_h['OBSNAME'] = h0['HIERARCH ESO OBS NAME']
        max_h['PROGID'] = h0['HIERARCH ESO OBS PROG ID']
        max_h['TARGNAME'] = h0['HIERARCH ESO OBS TARG NAME']
        for k in ['DATE-OBS', 'MJD-OBS']:
            max_h[k] = h0[k]
    
        max_h['EXPTIME'] = max_h['DIT']*max_h['NDIT']*max_h['NEXP']
        max_h['ZP'] = 25, 'Dummy zeropoint'
        max_h['PHOTFNU'] = 10**(-0.4*(25-8.90)), 'Dummy zeropoint'
        
        sci_scl = 1./max_h['DIT'] 
        relative_pscale = max_wcs.pscale/wcs_list[0].pscale
        wht_scl = max_h['DIT']**2*relative_pscale**-4
        
        pyfits.writeto('{0}-{1}-ks_drz_sci.fits'.format(ob_root, ext), data=_drzt[0]*sci_scl, header=max_h, clobber=True, output_verify='fix')
        pyfits.writeto('{0}-{1}-ks_drz_wht.fits'.format(ob_root, ext), data=_drzt[1]*wht_scl, header=max_h, clobber=True, output_verify='fix')
    
        cat = prep.make_SEP_catalog('{0}-{1}-ks'.format(ob_root, ext), threshold=3, column_case=str.upper)
        sn = cat['FLUX_APER_2'] / cat['FLUXERR_APER_2']
        clip = np.isfinite(sn) #& (cat['THRESH'] < 1e20)
        clip &= sn > 10
        clip &= cat['MASK_APER_2'] == 0
        cat = cat[clip]
        
        N = 150
        if len(cat) > 10:
            so = np.argsort(cat['MAG_AUTO'])[5:N+5]
        else:
            so = np.argsort(cat['MAG_AUTO'])[:]
            
        cat = cat[so]
        
        if assume_close > 0:
            idx, dr = radec_tab.match_to_catalog_sky(cat)
            
            if align_iter == 0:
                close_sep = assume_close
            else:
                close_sep = 0.2
                
            clip = dr.value < close_sep
            cat = cat[clip]
            
            # rev
            idx, dr = cat.match_to_catalog_sky(radec_tab)
            rclip = dr.value < close_sep
            if rclip.sum() > 150:
                _skip = (rclip.sum() // 150+1)
                rclip = np.where(rclip)[0][::_skip]
                
            prep.table_to_radec(radec_tab[rclip], 'tmp.radec')
            radec_i = 'tmp.radec'
        else:
            radec_i = radec
            rclip = np.ones(len(radec_tab), dtype=bool)
        
        cat.write('{0}-{1}-ks.cat.fits'.format(ob_root, ext), overwrite=True)
        
        if 'gaia' in radec:
            triangle_size = [10,3700]
        else:
            triangle_size = [5,200]
                        
        _res = prep.align_drizzled_image(root='{0}-{1}-ks'.format(ob_root, ext), triangle_size_limit=triangle_size, radec=radec_i, simple=False, NITER=3, triangle_ba_max=0.99, catalog_mask_pad=0.05, outlier_threshold=5, max_err_percentile=101, match_catalog_density=False, clip=align_clip, mag_limits=align_mag_limits, assume_close=(align_iter > 0))
        
        ######### Object masking
        if (align_iter == 0) & True:
            
            ################
            # Segmentation mask
            if seg_file is not None:
                if seg_file in [-1, 'None', 'none', 'NONE']:
                    seg_file = '{0}-{1}-ks_seg.fits'.format(ob_root, ext)
                    
                seg = pyfits.open(seg_file)
                seg_wcs = pywcs.WCS(seg[0].header)
                seg_data = seg[0].data

                msg = '{0}-{1}: Source segmentation mask = {2}'.format(ob_root, ext, seg_file)
                utils.log_comment(LOGFILE, msg, verbose=True)

                blt_mask = np.zeros(sci1.shape, dtype=bool)
                for i in range(NFILES):
                    if skip_list[i]:
                        continue
                        
                    #print('Source mask', sci_files[i], ext, i)
                    msg = '  Source Mask ({2}): {0}[{1}]'.format(sci_files[i], ext, i)
                    print(msg)
                    
                    sci_i = sci1[i,:]/flat1
                    sci_i[flat1 == 0] = 0
                    dq_i = dq & (flat1 != 0)

                    blt = utils.blot_nearest_exact(seg_data, seg_wcs, transform_wcs[i], verbose=True, stepsize=16)
                    dq_i &= (blt.flatten() <= 0)
                    blt_mask[i,:] |= dq_i
            else:
                blt_mask = np.zeros(sci1.shape, dtype=bool)
                for i in range(NFILES):
                    if skip_list[i]:
                        continue

                    blt_mask[i,:] = dq
                        
            ################ Redo flat
            if get_masked_flat:
                msg = '{0}-{1}: Redo masked flat'.format(ob_root, ext)
                utils.log_comment(LOGFILE, msg, verbose=True)
            
                scim = np.ma.masked_array(sci1, mask=~blt_mask)
                bg0m = np.ones(NFILES)
                for i in range(NFILES):
                    if skip_list[i]:
                        continue
                
                    bg0m[i] = np.ma.median(scim[i])
                    print('New median {0} : {1:.1f}'.format(i, bg0m[i]))
            
                # Now use median    
                med = np.ma.median(scim[~skip_list].T/bg0m[~skip_list], axis=1)
                med_n = (~scim.mask).sum(axis=0)
                med0 = np.ma.median(med) 
                flat2 = med/med0
                flat2.fill_value = 1.
            
                dq &= (~flat2.mask)
                # "flatten" the flat
                _fres = np.linalg.lstsq(_A[:,dq].T, flat2.filled()[dq], rcond='warn')
                flat_sm = _A.T.dot(_fres[0])
                
                h = pyfits.Header()

                if 0:
                    print('Flatten masked flat')
                    flat2 = flat2.filled()/flat_sm
                    h['FLATTEN'] = True
                else:
                    print('Don\'t flatten masked flat')
                    h['FLATTEN'] = False
                    flat2 = flat2.filled()
                    
                flat2[flat2 == 0] = 1
                
                # if os.path.exists(gradient_file):
                #     print('Use flat gradient file ', gradient_file)
                #     grad = pyfits.open(gradient_file)
                #     flat2 *= grad[0].data.flatten()
            
                # Write masked flat
                flat_file = '{0}-{1}.flat.masked.fits'.format(ob_root, ext)
            
                h['NFILES'] = NFILES
                for i, f in enumerate(sci_files):
                    h['FILE{0:04d}'.format(i)] = sci_files[i]

                hdu = pyfits.HDUList(pyfits.PrimaryHDU(data=flat2.reshape(sh),
                                     header=h))
                                             
                h = pyfits.Header()
                h['ISMASKED'] = True
                h['BKGORDER'] = bkg_order
                for j in range(len(_fres[0])):
                    h['BKG{0:03d}'.format(j)] = _fres[0][j]

                hdu.append(pyfits.ImageHDU(data=flat_sm.reshape(sh), header=h, name='POLY'))
                hdu.append(pyfits.ImageHDU(data=dq.reshape(sh)*1, name='DQ'))

                hdu.writeto(flat_file, overwrite=True)
                flat1 = flat2
            
            #############
            
            # Redo background subtraction
            sci_list = []
            wht_list = []
                        
            #blt_mask = np.zeros(sci1.shape, dtype=bool)
            for i in range(NFILES):
                
                if skip_list[i]:
                    sci_list.append(np.zeros(sh, dtype=np.float32))
                    wht_list.append(np.zeros(sh, dtype=np.float32))
                    
                    headers[i]['BKGORDER'] = bkg_order
                    for j in range(bkg_order**2):
                        headers[i]['BKG{0:03d}'.format(j)] = 0
                    
                    headers[i]['NMAD'] = -1
                    
                    continue
                    
                sci_i = sci1[i,:]/flat1
                dq_i = dq & (flat1 != 0)
                dq_i &= blt_mask[i,:]
                
                _bres = np.linalg.lstsq(_A[:,dq_i].T, sci_i[dq_i], rcond=-1)
                bg_i = _A.T.dot(_bres[0])
                
                msg = 'Masked background ({2:>2}): {0}[{1}] bg0={3:7.1f}'.format(sci_files[i], ext, i, _bres[0][0])
                utils.log_comment(LOGFILE, msg, verbose=True)
                
                headers[i]['BKGORDER'] = bkg_order
                for j in range(len(_bres[0])):
                    headers[i]['BKG{0:03d}'.format(j)] = _bres[0][j]
                    
                if ds9 is not None:
                    ds9.view(((sci_i - bg_i)*dq_i/100.).reshape(sh))
                
                sci_list.append((sci_i - bg_i).reshape(sh))
                nmad = utils.nmad(sci_list[-1][dqsh])
                headers[i]['NMAD'] = nmad, 'Robust NMAD standard deviation'
                
                wht_i =  np.cast[np.float32](dqsh/nmad**2)
                wht_list.append(wht_i)
        
        #radec = 'hff-j001408m3023_master.radec'
        #radec = 'gaia.radec'
            
        hdu = pyfits.open('{0}-{1}-ks_drz_sci.fits'.format(ob_root, ext), mode='update')
        hduw = pyfits.open('{0}-{1}-ks_drz_wht.fits'.format(ob_root, ext), mode='update')
        new_h = utils.to_header(_res[1])
        for k in new_h:
            hdu[0].header[k] = new_h[k]
            hduw[0].header[k] = new_h[k]
    
        hdu.flush()
        hduw.flush()
    
        cat = prep.make_SEP_catalog('{0}-{1}-ks'.format(ob_root, ext), threshold=2, column_case=str.upper)
        
        # Now use pointing catalog for second iteration
        use_exposures = False
                            
    # Save processed exposures
    for i in range(NFILES):        
        out_root = '{0}-{2:03d}-{1}'.format(ob_root, ext, primary_headers[i]['HIERARCH ESO TPL EXPNO'])
        if skip_list[i]:
            fp = open('Processed/{0}.skip'.format(out_root), 'w')
            fp.write(time.ctime())
            fp.close()
            
            continue
        
        print('Save data: Processed/{0}.sci.fits'.format(out_root))
        
        cat_list[i].write('Processed/{0}.cat.fits'.format(out_root), overwrite=True)
        
        wcs_header = utils.to_header(transform_wcs[i])
        out_header = headers[i].copy()
        out_header['ORIGFILE'] = sci_files[i]
        for k in wcs_header:
            out_header[k] = wcs_header[k]
        
        hkey = 'HIERARCH ESO '
        for k in ['DET DIT', 'DET NDIT', 'TPL NEXP', 'TPL EXPNO', 'OBS ID', 'OBS TARG NAME']: 
            out_header[hkey+k] = primary_headers[i][hkey+k]
                
        pyfits.writeto('Processed/{0}.sci.fits'.format(out_root), data=sci_list[i].reshape(sh).astype(np.float32), header=out_header, overwrite=True)
    
    