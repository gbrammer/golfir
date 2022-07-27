"""
Pipeline for generating new image segmentation
"""


import os
import inspect
import glob
import copy

import numpy as np
import scipy.ndimage as nd
import matplotlib.pyplot as plt

import astropy.io.fits as pyfits
import astropy.units as u
import astropy.wcs as pywcs

try:
    from grizli import utils, prep
except:
    pass
    
from tqdm import tqdm

phot_args = {'aper_segmask': True,
             'bkg_mask': None,
             'bkg_params': {'bh': 32, 'bw': 32, 
                            'fh': 5, 'fw': 5, 'pixel_scale': 0.1},
             'detection_background': True,
             'detection_filter': 'det',
             'detection_root': None,
             'get_all_filters': False,
             'output_root': None,
             'photometry_background': True,
             'rescale_weight': False,
             'run_detection': True,
             'threshold': 1.0,
             'use_bkg_err': False,
             'use_psf_filter': True, 
             'phot_apertures': prep.SEXTRACTOR_PHOT_APERTURES_ARCSEC}


GAUSSIAN_KERNEL = np.array([
[0.0049, 0.0213, 0.0513, 0.0687, 0.0513, 0.0213, 0.0049],
[0.0213, 0.0921, 0.2211, 0.296 , 0.2211, 0.0921, 0.0213],
[0.0513, 0.2211, 0.5307, 0.7105, 0.5307, 0.2211, 0.0513],
[0.0687, 0.296 , 0.7105, 0.9511, 0.7105, 0.296 , 0.0687],
[0.0513, 0.2211, 0.5307, 0.7105, 0.5307, 0.2211, 0.0513],
[0.0213, 0.0921, 0.2211, 0.296 , 0.2211, 0.0921, 0.0213],
[0.0049, 0.0213, 0.0513, 0.0687, 0.0513, 0.0213, 0.0049]])

F160W_KERNEL = np.array([
       [0.00036306, 0.0007211 , 0.0006885 , 0.00051918, 0.00060595,
        0.00044273, 0.00041596, 0.00042298, 0.0005109 , 0.0004347 ,
        0.00034758],
       [0.00049801, 0.00045392, 0.00043181, 0.00053092, 0.00097037,
        0.00124018, 0.00116621, 0.00075295, 0.00043989, 0.0004214 ,
        0.00075579],
       [0.00053497, 0.00041625, 0.00083392, 0.00219883, 0.00660488,
        0.00857547, 0.00503033, 0.00212257, 0.00075933, 0.00041211,
        0.00084069],
       [0.00043234, 0.00058102, 0.00164972, 0.00746829, 0.01723461,
        0.02033997, 0.01402212, 0.00828142, 0.00226408, 0.00056408,
        0.00080164],
       [0.00047637, 0.00072479, 0.00393787, 0.01261413, 0.03511766,
        0.07984465, 0.04050749, 0.01689382, 0.00597891, 0.00085294,
        0.00081502],
       [0.00054993, 0.00089698, 0.00678479, 0.01839382, 0.07425842,
        0.17333734, 0.07481   , 0.01872553, 0.00749245, 0.00089873,
        0.00055303],
       [0.00082199, 0.000887  , 0.00609313, 0.01695003, 0.04005031,
        0.07847298, 0.03627246, 0.0134794 , 0.00416312, 0.00083321,
        0.00047439],
       [0.0007945 , 0.00056122, 0.00236797, 0.00823446, 0.01374953,
        0.01950207, 0.0165822 , 0.00759804, 0.0017611 , 0.00061746,
        0.0004295 ],
       [0.0008201 , 0.00040671, 0.00073257, 0.00211902, 0.00480665,
        0.00771078, 0.00646757, 0.00208035, 0.0008136 , 0.00042777,
        0.00055984],
       [0.00076636, 0.00042482, 0.00044357, 0.00070887, 0.00106409,
        0.00114229, 0.00095213, 0.00050255, 0.00040967, 0.000425  ,
        0.00048569],
       [0.00035023, 0.00045955, 0.00051846, 0.00040028, 0.00039116,
        0.00045092, 0.00060395, 0.00052881, 0.00072058, 0.00070227,
        0.00034031]])
        
detection_params = {'minarea': 5, 
                    'filter_kernel': F160W_KERNEL, 
                    'filter_type': 'conv', 
                    'clean': False, 
                    'clean_param': 1,
                    'deblend_nthresh': 32,
                    'deblend_cont': 1.e-5}


def show_seg(seg, ds9, header=None, seed=1):
    """
    Show segmentation image in DS9 with randomized segments
    """
    from grizli.catalog import randomize_segmentation_labels
    ds9.view(randomize_segmentation_labels(seg, random_seed=seed)[1], 
          header=header)


def circle_footprint(scale=12, shrink=1):
    xarr = np.arange(scale+1) - scale//2
    yp, xp = np.meshgrid(xarr, xarr)
    R = np.sqrt((xp)**2+(yp)**2) < scale//2*shrink
    return R
    
    
def median_filter_circle(_data, scale=12, n_proc=10, cutout_size=256, **kwargs):
    """
    Run median (parallelized) median filter with a circular footprint
    
    Parameters
    ----------
    scale : int
        Filter diameter in pixels
    
    n_proc : bool
        Number of processors
    
    cutout_size : int
        Subimage size
    
    Returns
    -------
    filtered : array-like
        Filtered image
        
    """
    
    R = circle_footprint(scale=scale)
                
    filtered = utils.multiprocessing_ndfilter(_data, nd.median_filter, 
                                        filter_args=(), 
                                        footprint=R,
                                        n_proc=n_proc,
                                        cutout_size=cutout_size)

    return filtered


def match_catalog_layers(c0, c1, grow_radius=5, max_offset=5, low_layer=0, make_plot=False, sort_column='flux_iso'):
    """
    Match layered catalogs based on detection ellipses
    """
    from matplotlib.patches import Ellipse
    
    if make_plot:
        fig, ax = plt.subplots(1,1,figsize=(8,8))
    
    #c0['layer'] = large_layer - 1
    c1['valid'] = 2
    c0['hix'] = -1
    c1['lix'] = -1
    
    points = np.array([c0['x'], c0['y']]).T
    
    if make_plot:
        ax.scatter(c0['x'], c0['y'], color='k', alpha=0.2, marker='.')
    
    # Brightest source in level 0 also in level 1
    for i in np.argsort(c1[sort_column])[::-1]:
        center = (c1['x'][i], c1['y'][i])
        ell = Ellipse(center, c1['a_image'][i]*grow_radius, 
                              c1['b_image'][i]*grow_radius, 
                              angle=c1['theta_image'][i]/np.pi*180, 
                              facecolor='None', edgecolor='g',
                              alpha=0.5)
        
        mat = ell.contains_points(points) & (c0['layer'] == low_layer)
        if mat.sum() == 0:
            c1['valid'][i] = 0
        else:                
            mix = np.where(mat)[0]
            bix = np.argmax(c0[sort_column][mix])
            
            dx = c0['x'][mix[bix]] - c1['x'][i]
            dy = c0['y'][mix[bix]] - c1['y'][i]
            if np.sqrt(dx**2+dy**2) > max_offset:
                c1['valid'][i] = 1
                if make_plot:
                    ell.set_edgecolor('r')
                    ax.add_patch(ell)
                
                continue
                
            c0['layer'][mix[bix]] = c1['layer'][i]
            c0['hix'][mix[bix]] = i
            c1['lix'][i] = mix[bix]
            
            if make_plot:
                ax.add_patch(ell)
    
    if make_plot:
        ax.set_aspect(1)

        bix = c0['layer'] > low_layer
        ax.scatter(c0['x'][bix], c0['y'][bix], 
                   color='g', marker='.', alpha=0.6)


def analyze_image(data, err, seg, tab, athresh=3., 
                  robust=False, allow_recenter=False, 
                  prefix='', suffix='', grow=1, 
                  subtract_background=False, include_empty=False, 
                  pad=0, dilate=0, make_image_cols=True):
    """
    SEP/SExtractor analysis on arbitrary image
    
    Parameters
    ----------
    data : array
        Image array
    
    err : array
        RMS error array
    
    seg : array
        Segmentation array
    
    tab : `~astropy.table.Table`
        Table output from `sep.extract` where `id` corresponds to segments in 
        `seg`.  Requires at least columns of 
        ``id, xmin, xmax, ymin, ymax`` and ``x, y, flag`` if want to use 
        `robust` estimators
    
    athresh : float
        Analysis threshold
    
    prefix, suffix : str
        Prefix and suffix to add to output table column names
        
    Returns
    -------
    tab : `~astropy.table.Table`
        Table with columns
        ``id, x, y, x2, y2, xy, a, b, theta``
        ``flux, background, peak, xpeak, ypeak, npix``
        
    """
    from collections import OrderedDict
    import sep
    from grizli import utils, prep
    
    yp, xp = np.indices(data.shape) - 0.5*(grow == 2)
    
    # Output data
    new = OrderedDict()

    idcol = choose_column(tab, ['id','number'])
    ids = tab[idcol]
    
    new[idcol] = tab[idcol]
    
    for k in ['x','y','x2','y2','xy','a','b','theta','peak','flux','background']:
        if k in tab.colnames:
            new[k] = tab[k].copy()
        else:
            new[k] = np.zeros(len(tab), dtype=np.float32)
    
    for k in ['xpeak','ypeak','npix','flag']:
        if k in tab.colnames:
            new[k] = tab[k].copy()
        else:
            new[k] = np.zeros(len(tab), dtype=int)
        
    for id_i in tqdm(ids):
        ix = np.where(tab[idcol] == id_i)[0][0]
        
        xmin = tab['xmin'][ix]-1-pad
        ymin = tab['ymin'][ix]-1-pad
        
        slx = slice(xmin, tab['xmax'][ix]+pad+2)
        sly = slice(ymin, tab['ymax'][ix]+pad+2)
        
        seg_sl = seg[sly, slx] == id_i
        if include_empty:
            seg_sl |= seg[sly, slx] == 0
        
        if dilate > 0:
            seg_sl = nd.binary_dilation(seg_sl, iterations=dilate)
            
        if seg_sl.sum() == 0:
            new['flag'][ix] |= 1
            continue
                    
        if grow > 1:
            sh = seg_sl.shape
            seg_gr = np.zeros((sh[0]*grow, sh[1]*grow), dtype=bool)
            for i in range(grow):
                for j in range(grow):
                    seg_gr[i::grow, j::grow] |= seg_sl
            
            seg_sl = seg_gr
            
            xmin = xmin*grow
            ymin = ymin*grow

            slx = slice(xmin, (tab['xmax'][ix]+2+pad)*grow)
            sly = slice(ymin, (tab['ymax'][ix]+2+pad)*grow)
        
        if subtract_background:
            if subtract_background == 2:
                # Linear model
                x = xp[sly, slx] - xmin
                y = yp[sly, slx] - ymin
                A = np.array([x[~seg_sl]*0.+1, x[~seg_sl], y[~seg_sl]])
                b = data[sly,slx][~seg_sl]
                
                lsq = np.linalg.lstsq(A.T, b)
                back_level = lsq[0][0]
                A = np.array([x[seg_sl]*0.+1, x[seg_sl], y[seg_sl]]).T
                back_xy = A.dot(lsq[0])
            else:
                # Median
                back_level = np.median(data[sly, slx][~seg_sl])
                back_xy = back_level
        else:
            back_level = 0.
            back_xy = back_level

        dval = data[sly, slx][seg_sl] - back_xy
        ival = err[sly, slx][seg_sl]
        rv = dval.sum()
        
        imax = np.argmax(dval)
        peak = dval[imax]
        
        x = xp[sly, slx][seg_sl] - xmin
        y = yp[sly, slx][seg_sl] - ymin

        xpeak = x[imax] + xmin
        ypeak = y[imax] + ymin

        thresh_sl = (dval > athresh*ival) & (ival >= 0)

        new['npix'][ix] = thresh_sl.sum()
        new['background'][ix] = back_level
        
        if new['npix'][ix] == 0:
            new['flag'][ix] |= 2
            
            new['x'][ix] = np.nan
            new['y'][ix] = np.nan
            new['xpeak'][ix] = xpeak
            new['ypeak'][ix] = ypeak
            new['peak'][ix] = peak
            new['flux'][ix] = rv
            new['x2'][ix] = np.nan
            new['y2'][ix] = np.nan
            new['xy'] = np.nan
            new['a'][ix] = np.nan
            new['b'][ix] = np.nan
            new['theta'][ix] = np.nan
            continue
            
        cval = dval[thresh_sl]
        rv = cval.sum()
         
        x = x[thresh_sl]
        y = y[thresh_sl]
                
        mx = (x*cval).sum()
        my = (y*cval).sum()
        mx2 = (x*x*cval).sum()
        my2 = (y*y*cval).sum()
        mxy = (x*y*cval).sum()
        
        xm = mx/rv
        ym = my/rv
    
        xm2 = mx2/rv - xm**2
        ym2 = my2/rv - ym**2
        xym = mxy/rv - xm*ym
        
        if robust:
            if 'flag' in tab.colnames:
                flag = tab['flag'][ix] & sep.OBJ_MERGED
            else:
                flag = False
            
            if flag | (robust > 1):
                if allow_recenter:
                    xn = xm
                    yn = ym
                else:
                    xn = tab['x'][ix]-xmin
                    yn = tab['y'][ix]-ymin
                
                xm2 = mx2 / rv + xn*xn - 2*xm*xn
                ym2 = my2 / rv + yn*yn - 2*ym*yn
                xym = mxy / rv + xn*yn - xm*yn - xn*ym
                xm = xn
                ym = yn
            
        temp2 = xm2*ym2-xym*xym
        
        if temp2 < 0.00694:
            xm2 += 0.0833333
            ym2 += 0.0833333
            temp2 = xm2*ym2-xym*xym;
        
        temp = xm2 - ym2
        if np.abs(temp) > 0:
            theta = np.clip(np.arctan2(2.0*xym, temp)/2., 
                            -np.pi/2.+1.e-5, np.pi/2.-1.e-5)
        else:
            theta = np.pi/4
        
        temp = np.sqrt(0.25*temp*temp+xym*xym);
        pmy2 = pmx2 = 0.5*(xm2+ym2);
        pmx2 += temp
        pmy2 -= temp
        
        amaj = np.sqrt(pmx2)
        amin = np.sqrt(pmy2)
        
        new['x'][ix] = xm+xmin
        new['y'][ix] = ym+ymin
        new['xpeak'][ix] = xpeak
        new['ypeak'][ix] = ypeak
        new['peak'][ix] = peak
        new['flux'][ix] = rv
        new['x2'][ix] = xm2
        new['y2'][ix] = ym2
        new['xy'] = xym
        new['a'][ix] = amaj
        new['b'][ix] = amin
        new['theta'][ix] = theta
    
    new['flag'] |= ((~np.isfinite(new['a'])) | (new['a'] <= 0))*4
    new['flag'] |= ((~np.isfinite(new['b'])) | (new['b'] <= 0))*8
        
    newt = utils.GTable()
    for k in new:
        newt[f'{prefix}{k}{suffix}'] = new[k]
    
    if make_image_cols:
        newt['a_image'] = newt['a']
        newt['b_image'] = newt['b']
        newt['theta_image'] = newt['theta']
        newt['x_image'] = newt['x']+1
        newt['y_image'] = newt['y']+1
        
    return newt


def choose_column(tab, choices=['id','number'], raise_exception=True):
    """
    Find first column in a table from a list of choices
    
    Parameters
    ----------
    tab : `astropy.table.Table`
        Table
    
    choices : list of str
        Column choices
    
    raise_exception : bool
        Raise an exception if no columns found among `choices`.  Otherwise, 
        return empty string
    
    Returns
    -------
    col : str
        First element of `choices` found in columns of `tab`
        
    """

    for ch in choices:
        if ch in tab.colnames:
            return ch
    
    # If here, then no choices found
    if raise_exception:
        raise KeyError(f'No columns found from {choices}')
    else:
        return None


def switch_segments(seg, tab, new_id=None, verbose=True):
    """
    Switch segmentation labels
    
    Parameters
    ----------
    seg : array-like (int)
        Segmentation image
    
    tab : `astropy.table.Table`
        Catalog associated with `seg`
        
    new_id : array-like
        New ids (same length as `tab`).  If None, then use zeros.
    
    Notes
    -----
    `seg` is modified in-place.
    
    The id column of `tab` *is not* updated with the `new_id`
    
    """
    if verbose:
        _iter = tqdm(range(len(tab)))
    else:
        _iter = range(len(tab))
    
    id_col = choose_column(tab, ['id','number'])
    
    if new_id is None:
        new_id = np.zeros(len(tab), dtype=seg.dtype)
        
    for i in _iter:
        slx = slice(tab['xmin'][i], tab['xmax'][i]+1)
        sly = slice(tab['ymin'][i], tab['ymax'][i]+1)
        scut = seg[sly, slx]
        repl = scut == tab[id_col][i]
        scut[repl] = new_id[i]


def zero_out_segments(data, seg, tab, verbose=True):
    """
    Set pixels in data array to zero for objects in a seg/table
    
    Parameters
    ----------
    data : array-like
        Pixel data to modify.  *Modified in place*
    
    seg : array-like
        Segmentation image
    
    tab : table
        Associated table.  Needs ``xmin``, ``xmax``, ``ymin``, ``ymax`` 
        columns to define cutouts
    
    """
    if verbose:
        _iter = tqdm(range(len(tab)))
    else:
        _iter = range(len(tab))
    
    for i in _iter:
        slx = slice(tab['xmin'][i], tab['xmax'][i]+1)
        sly = slice(tab['ymin'][i], tab['ymax'][i]+1)
        scut = seg[sly, slx]
        repl = scut == tab['number'][i]
        data[repl] = 0


def remove_missing_ids(seg, tab, fill_value=0, verbose=True, logfile=None):
    """
    Merge table and segment image
    
    Parameters
    ----------
    seg : array-like
        Segmentation image
    
    tab : table
        Object table
    
    fill_value : int
        Value to insert into `seg` for segments without entries in `tab`
    
    Returns
    -------
    sfix, tfix : array-like, table
        Updated segmentation image and table with same number of objects
    
    """
    for id_col in ['id','number']:
        if id_col in tab.colnames:
            break
    
    seg_ids = np.unique(seg)[1:]
    s_list = []
    sfix = seg*1
    for sid in seg_ids:
        if sid not in tab[id_col]:
            _x = seg == sid
            sfix[_x] = fill_value
            s_list.append(sid)
    
    t_in_seg = np.array([i in seg_ids for i in tab[id_col]])
    msg = f'Remove {len(s_list):4} sources from seg not in table\n'
    msg += f'Remove {(~t_in_seg).sum():4} sources from table not in seg'
    if logfile is not None:
        utils.log_comment(logfile, msg, verbose=verbose)
    elif verbose:
        print(msg)
        
    return sfix, tab[t_in_seg]


def merge_close_sources(seg, tab, sep=0.5, verbose=True, logfile=None):
    """
    Merge sources in table separated by less than `sep` arcsec
    
    Notes
    -----
    `seg`, `tab` updated in place
    """
    tab['merged'] = 0
    try:
        ix, dr = tab.match_to_catalog_sky(tab, nthneighbor=2)
    except:
        print('Failed find neighbors, does `tab` have sky coords?')
        return False
    
    close = dr.value < sep
    merged = {}
    ii = np.arange(len(tab))
    
    ids = tab['id']
    
    for p, child in zip(ix[close], ii[close]):
        if child in merged:
            continue
        
        merged[child] = p
        if p in merged:
            parent = merged[p]
        else:
            merged[p] = p
            parent = p
        
        tab['merged'][parent] = 1
        tab['merged'][child] = -1
        
        #if verbose:
        msg = f'Merge IDs {ids[child]} > {ids[parent]}'
        if logfile is not None:
            utils.log_comment(logfile, msg, verbose=verbose)
        elif verbose:
            print(msg)
        
        _c, _p = child, parent
        tab['xmin'][_p] = np.minimum(tab['xmin'][_c], tab['xmin'][_p])
        tab['ymin'][_p] = np.minimum(tab['ymin'][_c], tab['ymin'][_p])
        tab['xmax'][_p] = np.maximum(tab['xmax'][_c], tab['xmax'][_p])
        tab['ymax'][_p] = np.maximum(tab['ymax'][_c], tab['ymax'][_p])
        tab['npix'][_p] += tab['ymax'][_c]
        
        switch_segments(seg, tab[child:child+1], [tab[parent]['id']], 
                        verbose=False)
    
    tab = tab[tab['merged'] >= 0]
    return True


RENAME_PROPERTIES = {'label':'id', 
                     'bbox-0':'ymin', 'bbox-1':'xmin',
                     'bbox-2':'ymax', 'bbox-3':'xmax',
                     'area':'npix',
                     'centroid-0':'x', 
                     'centroid-1':'y'}

def get_seg_limits(seg, intensity_image=None, properties=['label','bbox','area'], rename=RENAME_PROPERTIES):
    """
    Get bounding box of segments with `skimage.measure.regionprops_table`
    
    Parameters
    ----------
    seg : array-like
        Segmentation image
    
    intensity_image : array-like
        See `skimage.measure.regionprops_table`
        
    properties : list
        Properties to pass to `skimage.measure.regionprops_table`.
        
    rename : dict
        Dictionary for renaming properties columns in the output table
        
    Returns
    -------
    tab : `astropy.table.Table`
    """
    
    from skimage.measure import regionprops_table

    props = regionprops_table(seg, properties=properties)

    tab = utils.GTable()
    for k in props:
        if k in rename:
            tab[rename[k]] = props[k]
        else:
            tab[k] = props[k]
    
    return tab


def patch_ellipse_from_catalog(x, y, a, b, theta, **kwargs):
    """
    Make a `matplotlib.patches.Ellipse` patch from SourceExtractor
    parameters
    """
    from matplotlib.patches import Ellipse
    
    ell = Ellipse((x, y), a, b, angle=theta/np.pi*180, **kwargs)
    return ell
    
    
def shapely_ellipse_from_catalog(x, y, a, b, theta):
    """
    Make a `shapely.geometry.Polygon` patch from SourceExtractor
    parameters
    """
    from shapely.geometry import Point
    import shapely.affinity
    
    circ = Point([x,y]).buffer(1)
    ell = shapely.affinity.scale(circ, a/2, b/2)
    ellr = shapely.affinity.rotate(ell, theta/np.pi*180)
    return ellr


def make_charge_detection(root, ext='det', filters=['f160w','f140w','f125w','f110w','f105w','f814w','f850lp'], optical_kernel_sigma=1.3, scale_keyword='PHOTFLAM', run_catalog=False, mask_optical_weight=0.33, mask_ir_weight=-0.1, logfile=None, sep_bkg=[1.5, 16, 5], weight_pad=8, subtract_background=False, use_hst_kernel=True, parse_optical=True, **kwargs):
    """
    Make combined detection image for a given field
    """
    from sep import Background
    
    if logfile is None:
        logfile = f'{root}-{ext}_sm.log'
    
    frame = inspect.currentframe()
    utils.log_function_arguments(logfile, frame, func='make_charge_detection')

    from photutils.psf import IntegratedGaussianPRF
    from golfir.utils import convolve_helper
        
    nx = int(np.ceil(optical_kernel_sigma*5))
    xarr = np.arange(-nx, nx+1)
    yp, xp = np.meshgrid(xarr, xarr)
    prf = IntegratedGaussianPRF(sigma=optical_kernel_sigma)
    kernel = prf(xp, yp)
    kernel /= kernel.max()
    
    is_psf_match = False
    
    hst_kernel_file = os.path.join(os.path.dirname(__file__),
                                'data/psf/psf_kernel_f814w_f160w_50mas.fits')
                                   
    if os.path.exists(hst_kernel_file) & use_hst_kernel:
        msg = f'Use PSF-matching kernel {hst_kernel_file}'
        utils.log_comment(logfile, msg, verbose=True)
        
        imk = pyfits.open(hst_kernel_file)
        kernel = imk[0].data #/ imk[0].data.max()
        kernel = np.pad(kernel, pad_width=(0,1))
        is_psf_match = True
        
    files = []
    for filt in filters:
        _files = glob.glob(f'{root}-{filt}*_dr*sci.fits*')
        if '100mas' in root:
            _files += glob.glob(f"{root.replace('100mas','050mas')}-{filt}*_dr*sci.fits*")
            
        if len(_files) > 0:
            files.append(_files[0])
    
    for i, file in enumerate(files):
        sci = pyfits.open(file)
        wht = pyfits.open(file.replace('_sci.fits','_wht.fits'))
        
        if i == 0:
            if scale_keyword == 'fnu':
                ref_scale = sci[0].header['PHOTFLAM']/sci[0].header['PHOTPLAM']**2
            else:
                ref_scale = sci[0].header[scale_keyword]
                
            header = sci[0].header
            header['NCOMBINE'] = len(files)
            header['FSCALE'] = scale_keyword, 'Scale type'
            
            num = np.zeros_like(sci[0].data)
            den = np.zeros_like(num)
        
        header[f'FILE{i}'] = file
        header[f'FILT{i}'] = utils.get_hst_filter(sci[0].header), 'Filter'
        
        if scale_keyword == 'fnu':
            scale = sci[0].header['PHOTFLAM']/sci[0].header['PHOTPLAM']**2
        else:
            scale = sci[0].header[scale_keyword]
        
        scale /= ref_scale
        
        sh = sci[0].data.shape
        resize = sh[0] != num.shape[0]

        is_optical = ('_drc' in file) & parse_optical
        
        msg = f'{file} {scale:7.2f} optical={is_optical} resize={resize}'
        utils.log_comment(logfile, msg, verbose=True)
        
        bkg_img = 0.
        
        if sep_bkg is not None:
            mask = (sci[0].data*np.sqrt(wht[0].data) > sep_bkg[0]) 
            mask |= (wht[0].data <= 0)
            bh = sep_bkg[1]*1
            fw = sep_bkg[2]
            if resize:
                bh *= 2
                
            bkg = Background(sci[0].data*1, bh=bh, bw=bh, 
                             fw=fw, fh=fw, mask=mask)
            wht_i = 1/bkg.rms()**2
            wht_i *= wht[0].data > 0

            if subtract_background:
                bkg_img = bkg.back()
                
        else:
            wht_i = wht[0].data
        
        sci[0].data -= bkg_img
            
        if weight_pad > 0:
            msg = f'    weight_pad: {weight_pad}'
            utils.log_comment(logfile, msg, verbose=True)

            wht_mask = wht_i > 0
            wht_mask = nd.binary_dilation(wht_mask, 
                                     iterations=weight_pad*2**is_optical)
            wht_mask = nd.binary_erosion(wht_mask, 
                                     iterations=weight_pad*2*2**is_optical)
            wht_i *= wht_mask

        if is_optical:
            # ACS image.  Convolve and resample
            if is_psf_match:
                sm_num = convolve_helper(sci[0].data, kernel)
                sm_den = np.ones_like(sm_num)
            else:
                sm_num = convolve_helper(sci[0].data*wht_i, kernel)
                sm_den = convolve_helper(wht_i, kernel**2)

            sm_den[wht_i <= 0] = 0
            sm_avg = sm_num/sm_den
            sm_avg[sm_den <= 0] = 0
            
            if mask_optical_weight > 0:
                valid = wht_i > 0
                thresh = mask_optical_weight*np.median(wht_i[valid])
                wht_mask = wht_i > thresh
                wht_mask = nd.binary_erosion(wht_mask, iterations=2)
                wht_mask = nd.binary_dilation(wht_mask, iterations=2)
                
                frac = (valid & wht_mask).sum() / valid.sum() * 100
                msg = f'    Mask wht: {mask_optical_weight} valid={frac:.1f}%'
                utils.log_comment(logfile, msg, verbose=True)
                
                wht_i *= wht_mask
                sm_avg *= wht_mask
                
            if resize:
                for i in range(2):
                    for j in range(2):
                        num += (sm_avg*wht_i)[i::2, j::2]/(scale*4)
                        den += (wht_i)[i::2, j::2]/(scale*4)**2
            else:
                num += sm_num/scale
                den += sm_den/scale**2
        else:
            if mask_ir_weight > 0:
                valid = wht_i > 0
                thresh = mask_ir_weight*np.median(wht_i[valid])
                wht_mask = wht_i > thresh
                wht_mask = nd.binary_erosion(wht_mask, iterations=2)
                wht_mask = nd.binary_dilation(wht_mask, iterations=2)
                
                frac = (valid & wht_mask).sum() / valid.sum() * 100
                msg = f'    Mask wht: {mask_ir_weight} valid={frac:.1f}%'
                utils.log_comment(logfile, msg, verbose=True)
                
                wht_i *= wht_mask
                            
            num += sci[0].data*wht_i/scale
            den += wht_i/scale**2
        
    avg = num/den
    avg[den <= 0] = 0
    avg_wht = den
    
    pyfits.writeto(f'{root}-{ext}_drz_sci.fits', data=avg, header=header, 
                   overwrite=True)
    pyfits.writeto(f'{root}-{ext}_drz_wht.fits', data=avg_wht, header=header, 
                   overwrite=True)
    
    if run_catalog:
        fdet = FilterDetection(root, filter=ext)
        fdet.pipeline(**kwargs)
        return fdet
    
    else:
        return True


def set_iso_total_flux(phot):
    """
    Use `flux_iso` as the total flux
    
    Calculate an effective "aperture" of the iso segment as
    
        >>> r_iso = np.sqrt(phot['area_iso']/np.pi/q)
    
    and for each aperture, use the aperture flux itself as the total if that
    aperture is larger than `r_iso`.
    
    """
    
    from grizli import prep
    #phot = utils.read_catalog(phot_file)
    
    # Which aperture to use, or loop over them
    for ix in range(10):
        if f'APER_{ix}' not in phot.meta:
            continue
    
        total_flux = phot['flux_iso']*1

        ap_size = phot.meta[f'APER_{ix}']/2
        print(f'Aperture {ix} D={ap_size*2*0.1:.1f}"')

        # Rough SMA of the iso footprint, i.e., the segmented pixels
        if 'a' in phot.colnames:
            q = phot['b']/phot['a']
        else:
            q = phot['b_image']/phot['a_image']
            
        riso = np.sqrt(phot['area_iso']/np.pi/q)

        # Force aperture measurement for very small iso segments
        small = riso < ap_size
        total_flux[small] = phot[f'flux_aper_{ix}'][small]*1
        phot[f'r_iso_{ix}'] = np.maximum(riso, ap_size)
        phot[f'r_iso_{ix}'].description = 'Isophotal aperture size, pix'
        
        ap_radius = phot[f'r_iso_{ix}']
        
        apcorr = np.clip(total_flux / phot[f'flux_aper_{ix}'], 1, 10)

        # Encircled energy correction, depends on band
        bands = []
        for k in phot.meta:
            if k.endswith('_ZP'):
                bands.append(k.split('_')[0].lower())
        
        phot.meta['ISOTOTAL'] = True, 'Total flux defined by flux_iso, area_iso'
        
        #b = 'f160w'
        for b in bands:
            zp_b, aper_ee = prep.get_hst_aperture_correction(b, 
                                                raper=ap_radius*0.1, rmax=5.)

            totcorr = apcorr/aper_ee
            
            phot[f'{b}_eecorr_{ix}'] = 1./aper_ee
            phot[f'{b}_eecorr_{ix}'].description = 'Encircled energy correction for flux outside the total (iso) aperture'
            
            phot[f'{b}_tot_{ix}'] = phot[f'{b}_flux_aper_{ix}']*totcorr
            phot[f'{b}_tot_{ix}'].unit = phot[f'{b}_flux_aper_{ix}'].unit
            phot[f'{b}_etot_{ix}'] = phot[f'{b}_fluxerr_aper_{ix}']*totcorr
            phot[f'{b}_etot_{ix}'].unit = phot[f'{b}_flux_aper_{ix}'].unit
            
            phot[f'{b}_tot_{ix}'].description = 'Total flux corrected for aperture radii and flux outside the total (iso) aperture'
            phot[f'{b}_etot_{ix}'].description = 'Total uncertainty corrected for aperture radii and flux outside the total (iso) aperture'

    return phot
    
class FilterDetection(object):
    """
    Source detection on median-filtered images
    """
    def __init__(self, root='hff-j024004m0136', filter='f160w'):
        """
        Parameters
        ----------
        root : str
            Rootname for CHArGE products
        
        filter : str
            Filter to use as detection image, i.e., 
            ``'{root}-{filter}_drz_sci.fits'``
        
        """
        
        self.root = root
        self.filter = filter
        self.logfile = f'{root}-{filter}_sm.log'
        
        self.scales = [0,0]
        
        self.im = pyfits.open(f'{root}-{filter}_drz_sci.fits')
        self.imw = pyfits.open(f'{root}-{filter}_drz_wht.fits')
        self.data = self.im[0].data.byteswap().newbyteorder()
        self.wdata = self.imw[0].data.byteswap().newbyteorder()
        msk = ~np.isfinite(self.data)
        self.data[msk] = 0
        self.wdata[msk] = 0
        
        self.header = self.im[0].header
        self.wcs = pywcs.WCS(self.header)
        self.pixel_scale = utils.get_wcs_pscale(self.wcs)
        
        self.sh = self.im[0].data.shape
        
        self.filtered = {}
        self.wfiltered = {}
        
        self.watershed_thresholds = [0.3, 1.8]


    def smooth_scales(self, scales=[16,48], **kwargs):
        """
        Run `median_filter` on multiple scales
        
        Parameters
        ----------
        scales : [int, int]
            Filter scales
        
        Notes
        -----
        Populates `filtered` and `wfiltered` attributes
        
        """        
        for i, scale in enumerate(scales):
            if self.scales[i] == scale:
                continue
            
            self.scales[i] = scales[i]
             
            msg = f'Filter {self.root}-{self.filter} on scale {scale}'
            utils.log_comment(self.logfile, msg, verbose=True)
            
            self.filtered[i] = median_filter_circle(self.data, 
                                                        scale=scale, **kwargs)
            self.wfiltered[i] = median_filter_circle(self.wdata, 
                                                         scale=scale,**kwargs)
                            
        
    def run_detection(self, thresholds=[1.3, 1.0, 3.0], get_background=False, err_scale=0.8, large_scale_weight=0.5, **kwargs):
        """
        Run source detection on filtered images
        
        Parameters
        ----------
        thresholds : [float, float, float]
            Detection thresholds on filter layers
        
        Notes
        -----
        Filter layers:
        
            0. `data - filtered[0]` for fine structure
            1. `filtered[0] - filtered[1]` for moderately bright sources
            2. `filtered[1]`, `wfiltered[1]` weights for bright, extended 
                sources
        
        Detection catalogs put in `cat0`, `cat1`, `cat2` attributes
        
        """
        frame = inspect.currentframe()
        utils.log_function_arguments(self.logfile, frame, 
                                    func='FilterDetection.run_detection')
        
        ### FITS files
        pyfits.writeto(f'{self.root}-{self.filter}-0_drz_sci.fits', 
                       data=self.data-self.filtered[0],
                       header=self.header, overwrite=True)
        
        pyfits.writeto(f'{self.root}-{self.filter}-1_drz_sci.fits', 
                       data=self.filtered[0]-self.filtered[1],
                       header=self.header, overwrite=True)
        
        # pyfits.writeto(f'{self.root}-{self.filter}-2_drz_sci.fits', 
        #                data=self.filtered[1],
        #                header=self.header, overwrite=True)
        
        pyfits.writeto(f'{self.root}-{self.filter}-0_drz_wht.fits', 
                       data=self.wdata,
                       header=self.header, overwrite=True)
        
        pyfits.writeto(f'{self.root}-{self.filter}-1_drz_wht.fits', 
                       data=self.wdata,
                       header=self.header, overwrite=True)
        
        # pyfits.writeto(f'{self.root}-{self.filter}-2_drz_wht.fits', 
        #                data=self.wfiltered[1],
        #                header=self.header, overwrite=True)
        
        ### Run detection
        #bkg_params={'bw': 64, 'bh': 64, 'fw': 3, 'fh': 3}
        bkg_params = phot_args['bkg_params']
        
        args = dict(detection_params=copy.deepcopy(detection_params),
                    bkg_params=copy.copy(phot_args['bkg_params']), 
                    get_background=get_background,
                    column_case=str.lower,
                    err_scale=-np.inf,
                    pixel_scale=self.pixel_scale, rescale_weight=True,
                    phot_apertures=copy.copy(phot_args['phot_apertures']))
        
        ## First step: data - filter1
        args['rescale_weight'] = True
        args['err_scale'] = err_scale
                
        self.cat0 = prep.make_SEP_catalog(f'{self.root}-{self.filter}-0',
                                          threshold=thresholds[0], **args)
        
        # Trim bad
        #trim = (self.cat0['flux_aper_1'] / self.cat0['fluxerr_aper_1']) < 1.
        #self.cat0 = self.cat0[~trim]
        
        ## Next step: filter1 - filter2
        args['get_background'] = False
        args['detection_params']['minarea'] = (8**2)
        args['rescale_weight'] = True
        args['err_scale'] = self.cat0.meta['ERR_SCALE'][0]*large_scale_weight

        self.cat1 = prep.make_SEP_catalog(f'{self.root}-{self.filter}-1',
                                          threshold=thresholds[1], **args)
                
        ##############
        
        #trim = (self.cat1['flux_aper_1'] / self.cat1['fluxerr_aper_1']) < 1.
        #self.cat1 = self.cat1[~trim]
        
        ## Last step: filter2
        # if 0:
        #     args['detection_params']['minarea'] = (24**2)
        #     args['detection_params']['deblend_cont'] = 1.e-3
        #     args['get_background'] = True
        #     args['rescale_weight'] = True
        #     args['err_scale'] = self.cat0.meta['ERR_SCALE'][0]*large_scale_weight
        #     args['bkg_params'] = {'bw': 256, 'bh': 256, 'fw': 3, 'fh': 3}
        #     self.cat2 = prep.make_SEP_catalog(f'{self.root}-{self.filter}-2',
        #                                       threshold=thresholds[2], **args)
        
        #trim = (self.cat2['flux_aper_1'] / self.cat2['fluxerr_aper_1']) < 1.
        #self.cat2 = self.cat2[~trim]


    def combine_catalogs(self, max_offsets=[25,50], make_plot=False, sort_column='flux_iso', large_qmin=0.2, grow_radius=3, sn_layer1=500, **kwargs):
        
        from matplotlib.patches import Ellipse
        import astropy.table
        
        cat0 = utils.read_catalog(f'{self.root}-{self.filter}-0.cat.fits')
        cat1 = utils.read_catalog(f'{self.root}-{self.filter}-1.cat.fits')
        
        prep.table_to_regions(cat0, f'{self.root}-{self.filter}-0.reg',
                                      use_world=False, scale_major=3,
                                      use_ellipse=True, 
                                      header='global color=pink')

        prep.table_to_regions(cat1, f'{self.root}-{self.filter}-1.reg',
                                      use_world=False, scale_major=3,
                                      use_ellipse=True,
                                      header='global color=red')

        # if 0:
        #     cat2 = utils.read_catalog(f'{self.root}-{self.filter}-2.cat.fits')
        #     prep.table_to_regions(cat2, f'{self.root}-{self.filter}-2.reg',
        #                               use_world=False, scale_major=3,
        #                               use_ellipse=True,
        #                               header='global color=orange')
        
        cat0['layer'] = 0
        cat1['layer'] = 1
        
        if 0:
            pass
            # cat2['layer'] = 2
            #         
            # large_q = cat2['b_image']/cat2['a_image']
            # clip = large_q < large_qmin
            # cat2 = cat2[~clip]
            # cat2['new_id'] = np.arange(len(cat2), dtype=int)+1
            # 
            # xp2 = np.round(cat2['x']).astype(int)
            # yp2 = np.round(cat2['y']).astype(int)
            # 
            # n2 = len(cat2)
        else:
            n2 = 0
            
        cat1['new_id'] = n2 + np.arange(len(cat1), dtype=int)+1
        cat0['new_id'] = n2 + len(cat1)
        cat0['new_id'] += np.arange(len(cat0), dtype=int)+1
        
        wdata = (self.filtered[0] - self.filtered[1])
        xp0 = np.round(cat0['x']).astype(int)
        yp0 = np.round(cat0['y']).astype(int)
        xp1 = np.round(cat1['x']).astype(int)
        yp1 = np.round(cat1['y']).astype(int)
        sn1 = wdata*np.sqrt(self.wfiltered[1])
        #lost1 = sn1[yp1, xp1] < self.watershed_thresholds[0]

        if 0:
            wdata = (self.filtered[1])
            sn2 = wdata*np.sqrt(self.wfiltered[1])
        else:
            wdata = self.filtered[0]
            sn2 = wdata*np.sqrt(self.wdata)
            
        #lost2 = sn2[yp2, xp2] < self.watershed_thresholds[1]
        
        if 0:
            pass
            # ### Objects in layer1 inherited from layer2
            # match_catalog_layers(cat1, cat2, max_offset=max_offsets[1],
            #                      low_layer=1, grow_radius=grow_radius, 
            #                      sort_column=sort_column,
            #                      make_plot=make_plot)
            #         
            # thr = sn2[yp1, xp1] > self.watershed_thresholds[1]
            # layer12 = (cat1['layer'] == 2) & (thr)
            # cat1['layer'][~layer12] = 1
            #         
            # # pull ellipse parameters
            # for c in ['a_image','b_image','theta_image',
            #           'xmin','xmax','ymin','ymax','number','new_id']:
            #     cat1[c][layer12] = cat2[c][cat1['hix'][layer12]]
        else:
            # Put brightest objects from layer 1 in layer 2
            sn_auto = cat1['flux_aper_4']/cat1['fluxerr_aper_4']
            sn_clip = sn_auto > sn_layer1
            
            if sn_clip.sum() == 0:
                sn_clip = sn_auto == sn_auto.max()
                
            # remove large elongated
            large_q = cat1['b_image']/cat1['a_image']
            clip = large_q < large_qmin
            remove = (sn_clip & clip)
            cat1 = cat1[~remove]
            xp1 = np.round(cat1['x']).astype(int)
            yp1 = np.round(cat1['y']).astype(int)
        
            sn_auto = cat1['flux_aper_4']/cat1['fluxerr_aper_4']
            sn_clip = sn_auto > sn_layer1
            if sn_clip.sum() == 0:
                sn_clip = sn_auto == sn_auto.max()
        
            so = np.argsort(sn_auto[sn_clip])[::-1]
            acirc = np.sqrt(cat1['a_image']*cat1['b_image'])
            so = np.argsort(acirc[sn_clip])[::-1]
        
            snso = sn_auto[sn_clip][so]
            scat1 = cat1[sn_clip][so]
            scale_major = 3*10**np.clip(np.log10(snso/sn_layer1), 0, 0.5)
        
            x, y = scat1['x'], scat1['y']
            a = scat1['a_image']*scale_major
            b = scat1['b_image']*scale_major
            th = scat1['theta_image']
        
            keep = np.zeros(len(so), dtype=bool)
        
            for i in range(len(so)):
                p_i = shapely_ellipse_from_catalog(x[i], y[i], 
                                                    a[i], b[i], th[i])
            
                if i == 0:
                    keep[i] = True
                    full_poly = p_i
                else:
                    ol = full_poly.intersection(p_i)
                    keep[i] = ol.area < 0.01
                    full_poly = full_poly.union(p_i)
                
                if __name__ == '__main__':
                    from descartes import PolygonPatch
                    ax = plt.gca()
                    if keep[i]:
                        ax.add_patch(PolygonPatch(p_i, alpha=0.2, 
                                     color='k'))
                    else:
                        ax.add_patch(PolygonPatch(p_i, alpha=0.2, 
                                     color='r'))
                    
        
            if keep.sum() > 0:
                clip_ix = np.where(sn_clip)[0][so][keep]
        
                msg = f'{self.root}  Set {keep.sum()} sources to layer2 with S/N > {sn_layer1}'
                utils.log_comment(self.logfile, msg, verbose=True)
        
                cat1['layer'][clip_ix] = 2
            
        ### Objects in layer0 inherited from layer > 0
        match_catalog_layers(cat0, cat1, max_offset=max_offsets[0],
                             low_layer=0, grow_radius=grow_radius, 
                             sort_column=sort_column,
                             make_plot=make_plot)
        
        thr = sn1[yp0, xp0] > self.watershed_thresholds[0]
        layer01 = (cat0['layer'] > 0) & (thr)
        cat0['layer'][~layer01] = 0
        
        # pull ellipse parameters
        for c in ['a_image','b_image','theta_image',
                  'xmin','xmax','ymin','ymax','number','new_id']:
            cat0[c][layer01] = cat1[c][cat0['hix'][layer01]]
                
        ### unmatched objects in layer0 to layer2
        if 0:
            pass
            # xcat1 = cat0[cat0['layer'] > 0]
            # ix0 = cat0['layer'] == 0
            # xcat0 = cat0[ix0]
            # cat2['used'] = False
            # iu = cat1['hix'][cat0['hix']]
            # cat2['used'][iu[iu > 0]] = True
            # ix2 = ~cat2['used']
            # xcat2 = cat2[ix2]
            #         
            # match_catalog_layers(xcat0, xcat2, max_offset=max_offsets[0],
            #                      low_layer=0, grow_radius=grow_radius, 
            #                      sort_column=sort_column, 
            #                      make_plot=make_plot)
            #         
            # thr = sn2[yp0[ix0], xp0[ix0]] > self.watershed_thresholds[1]
            # layer02 = (xcat0['layer'] > 0) & (thr)
            # xcat0['layer'][layer02] = 2
            #         
            # for c in ['a_image','b_image','theta_image',
            #           'xmin','xmax','ymin','ymax','number','new_id']:
            #     xcat0[c][layer02] = xcat2[c][xcat0['hix'][layer02]]
            #         
            # for c in cat0.colnames:
            #     cat0[c][ix0] = xcat0[c]
        
        # Leftover sources in layer1
        add1 = sn1[yp1,xp1] > self.watershed_thresholds[0]
        add1 &= (cat1['lix'] < 0) & (cat1['layer'] == 1) & (cat1['valid'] == 0)
        
        if 0:
            extra = []
            for i, layer in enumerate(cat0['layer']):
                if layer == 0:
                    extra.append('color=pink')
                elif layer == 1:
                    extra.append('color=red')
                else:
                    extra.append('color=orange')
                
            prep.table_to_regions(cat0, '/tmp/cat0.reg',
                                          use_world=False, scale_major=3,
                                          use_ellipse=True, 
                                          extra=extra)
            #
            extra = []
            for i, layer in enumerate(cat1['layer'][add1]):
                if layer == 0:
                    extra.append('color=green')
                elif layer == 1:
                    extra.append('color=cyan')
                else:
                    extra.append('color=magenta')
                
            prep.table_to_regions(cat1[add1], '/tmp/cat1.reg',
                                          use_world=False, scale_major=3,
                                          use_ellipse=True, 
                                          extra=extra)
        
        det_cat = utils.GTable(astropy.table.vstack([cat0, cat1[add1]]))
        un, ix = np.unique(det_cat['new_id'], return_index=True)
        det_cat = det_cat[ix]
        
        if 0:
            xi = np.round(det_cat['x']).astype(int)
            yi = np.round(det_cat['y']).astype(int)
            xy = xi + yi*(xi.max()+1)
            v, uix = np.unique(xy, return_index=True)
            det_cat = det_cat[uix]
        
        for k in ['number','id']:
            det_cat[k] = det_cat['new_id']
            
        vs, cts = np.unique(det_cat['layer'], return_counts=True)
        for v, ct in zip(vs, cts):
            msg = f'{self.root}  layer={v}  N={ct}'
            utils.log_comment(self.logfile, msg, verbose=True)
        
        # Coords
        rd = self.wcs.all_pix2world(det_cat['x'], det_cat['y'], 0)
        det_cat['ra'], det_cat['dec'] = rd
        
        so = np.argsort(det_cat['id'])
        
        return det_cat[so]
        
        
    def xcombine_catalogs(self, sep1=0.5, sep2=2.0, **kwargs):
        """
        Combine catalogs taking new sources detected at each level
        
        Parameters
        ----------
        sep1 : float
            Separation in arcseconds that defines "new" sources detected in 
            layer 0
        
        sep2 : float
            Separation in arcseconds that defines new sources detected in 
            layer 1
        
        Returns
        -------
        det_cat : `astropy.table.Table`
            Combined detection table
        
        """
        frame = inspect.currentframe()
        utils.log_function_arguments(self.logfile, frame, 
                                    func='FilterDetection.combine_catalogs')
                                    
        import astropy.table
        
        # Find things that won't pass threshold later so they may be caught
        # at lower layer
        wdata = (self.filtered[0] - self.filtered[1])
        xp = np.round(self.cat1['x']).astype(int)
        yp = np.round(self.cat1['y']).astype(int)
        sn1 = wdata*np.sqrt(self.wfiltered[1])
        lost1 = sn1[yp, xp] < self.watershed_thresholds[0]

        xp = np.round(self.cat2['x']).astype(int)
        yp = np.round(self.cat2['y']).astype(int)
        wdata = (self.filtered[1])
        sn2 = wdata*np.sqrt(self.wfiltered[1])
        lost2 = sn2[yp, xp] < self.watershed_thresholds[1]
        
        # bad sources?
        bad = (self.cat0['thresh'] <= 0) | (self.cat0['thresh'] > 1.e10)
        trim = (self.cat0['flux_aper_1'] / self.cat0['fluxerr_aper_1']) < 1.
        bad |= trim
        self.cat0 = self.cat0[~bad]
        msg = f'Remove {bad.sum()} bad threshold sources from cat0'
        utils.log_comment(self.logfile, msg, verbose=True)

        bad = (self.cat1['thresh'] <= 0) | (self.cat1['thresh'] > 1.e10)
        trim = (self.cat1['flux_aper_1'] / self.cat1['fluxerr_aper_1']) < 1.
        bad |= trim | lost1
        self.cat1 = self.cat1[~bad]
        msg = f'Remove {bad.sum()} bad threshold sources from cat1'
        utils.log_comment(self.logfile, msg, verbose=True)

        bad = (self.cat2['thresh'] <= 0) | (self.cat2['thresh'] > 1.e10)
        trim = (self.cat2['flux_aper_1'] / self.cat2['fluxerr_aper_1']) < 1.
        bad |= trim | lost2
        self.cat2 = self.cat2[~bad]
        msg = f'Remove {bad.sum()} bad threshold sources from cat2'
        utils.log_comment(self.logfile, msg, verbose=True)
        
        ix, dr = self.cat0.match_to_catalog_sky(self.cat1)
        so = np.argsort(dr)
        ok = (dr.value < sep1) 
        
        self.cat0['new'] = True
        self.cat0['new'][ix[ok]] = False
        self.cat0['layer'] = 0
        self.cat0['dr'] = 0.
        self.cat1['dr'] = dr.value
        
        prep.table_to_regions(self.cat0,
                              f'{self.root}-{self.filter}-layer0.reg',
                              use_world=False, use_ellipse=False, 
                              header='global color=pink', size=5)

        prep.table_to_regions(self.cat1,
                              f'{self.root}-{self.filter}-layer1.reg',
                              use_world=False, use_ellipse=False, 
                              header='global color=red', size=5)
        
        # Put coords from lower level 
        for c in ['x_image','y_image','x_world','y_world','x','y','ra','dec']:
            used = []
            for i,j in enumerate(so):
                k = ix[j]
                if not ok[j]:
                    continue

                if k in used:
                    continue
                else:
                    used.append(k)

                self.cat1[c][j] = self.cat0[c][k]*1

        # Next level
        ix, dr = self.cat1.match_to_catalog_sky(self.cat2)
        so = np.argsort(dr)
        self.cat2['layer'] = 2
        self.cat2['dr'] = dr.value
        ok = (dr.value < sep2)
        prep.table_to_regions(self.cat2, 
                              f'{self.root}-{self.filter}-layer2.reg',
                              use_world=False, use_ellipse=False, 
                              header='global color=orange', size=5)

        for c in ['x_image','y_image','x','y','x_world','y_world','ra','dec']:
            used = []
            for i,j in enumerate(so):
                k = ix[j]
                if not ok[j]:
                    continue

                if k in used:
                    continue
                else:
                    used.append(k)

                self.cat2[c][j] = self.cat1[c][k]*1

        self.cat1['new'] = True
        self.cat1['new'][ix[ok]] = False
        self.cat1['layer'] = 1

        self.cat2['new_id'] = np.arange(len(self.cat2), dtype=int)+1
        self.cat1['new_id'] = np.arange(len(self.cat1), dtype=int)+1
        self.cat1['new_id'] += self.cat2['new_id'].max()
        
        self.cat0['new_id'] = np.arange(len(self.cat0), dtype=int)+1
        self.cat0['new_id'] += self.cat1['new_id'].max()

        det_cat = utils.GTable(astropy.table.vstack([self.cat2, 
                                     self.cat1[self.cat1['new']], 
                                     self.cat0[self.cat0['new']]]))
        
        if 0:
            xi = np.round(det_cat['x']).astype(int)
            yi = np.round(det_cat['y']).astype(int)
            xy = xi + yi*(xi.max()+1)
            v, uix = np.unique(xy, return_index=True)
            det_cat = det_cat[uix]
        
        for k in ['number','id']:
            det_cat[k] = det_cat['new_id']
            
        vs, cts = np.unique(det_cat['layer'], return_counts=True)
        for v, ct in zip(vs, cts):
            msg = f'{self.root}  layer={v}  N={ct}'
            utils.log_comment(self.logfile, msg, verbose=True)
        
        # Coords
        rd = self.wcs.all_pix2world(det_cat['x'], det_cat['y'], 0)
        det_cat['ra'], det_cat['dec'] = rd
        
        return det_cat


    def combine_segments(self, expand_smallest=1, merge_sep=-0.5, **kwargs):
        """
        Combine segmentation image from sequence of catalogs
        
        Parameters
        ----------
        expand_smallest : int
            Grow layer 0 segments by this factor
        
        merge_sep : float
            If > 0, then merge sources closer than this separation in arcsec
        
        Notes
        -----
        Sets `det_seg` and modifies `det_cat` attributes
        
        """
        frame = inspect.currentframe()
        utils.log_function_arguments(self.logfile, frame, 
                                    func='FilterDetection.combine_segments')
                                    
        from skimage.segmentation import expand_labels
        
        s0 = pyfits.open(f'{self.root}-{self.filter}-0_seg.fits')[0].data
        s1 = pyfits.open(f'{self.root}-{self.filter}-1_seg.fits')[0].data
        s2 = pyfits.open(f'{self.root}-{self.filter}-2_seg.fits')[0].data
        
        s0, self.cat0 = remove_missing_ids(s0, self.cat0, 
                           fill_value=0, verbose=True, logfile=None)
        
        s1, self.cat1 = remove_missing_ids(s1, self.cat1, 
                           fill_value=0, verbose=True, logfile=None)
        
        s2, self.cat2 = remove_missing_ids(s2, self.cat2, 
                           fill_value=0, verbose=True, logfile=None)
        
        d00 = pyfits.open(f'{self.root}-{self.filter}-0_drz_sci.fits')[0].data
        w00 = pyfits.open(f'{self.root}-{self.filter}-0_drz_wht.fits')[0].data
        d12 = pyfits.open(f'{self.root}-{self.filter}-1_drz_sci.fits')[0].data
        d48 = pyfits.open(f'{self.root}-{self.filter}-2_drz_sci.fits')[0].data
        
        ###### First step
        # remove segments of sources not "new" in 0th level
        switch_segments(s0, self.cat0[~self.cat0['new']],
                        np.zeros((~self.cat0['new']).sum(), dtype=int))
        
        # rename to new ids
        switch_segments(s0, self.cat0[self.cat0['new']], 
                        self.cat0['new_id'][self.cat0['new']])
        
        if expand_smallest > 0:
            msg = f'Expand smallest labels distance={expand_smallest}'
            utils.log_comment(self.logfile, msg, verbose=True)
            
            s0 = expand_labels(s0, distance=expand_smallest)
            
        ###### Second step
        switch_segments(s1, self.cat1[~self.cat1['new']],
                        np.zeros((~self.cat1['new']).sum(), dtype=int))
        switch_segments(s1, self.cat1[self.cat1['new']],
                        self.cat1['new_id'][self.cat1['new']])
        # Insert
        s0[s0 == 0] += s1[s0 == 0]
        
        ###### Last step
        switch_segments(s2, self.cat2, self.cat2['new_id'])
        s0[s0 == 0] += s2[s0 == 0]
        
        ##### Clean up segment image and catalog
        _ = remove_missing_ids(s0, self.det_cat, fill_value=0, verbose=True)
        s0, self.det_cat = _
        
        if merge_sep > 0:
            merge_close_sources(s0, self.det_cat, sep=merge_sep, 
                                verbose=True, logfile=self.logfile)
        
        self.det_seg = s0
    
    
    @staticmethod
    def reanalyze_image(data, err, seg, cat, data_bkg=None, ZP=23.9, autoparams=[2.5, 0.35*u.arcsec, 0, 5], flux_radii=[0.2, 0.5, 0.9], min_a=0.35, analyze_robust=2, analyze_recenter=False, pixel_scale=0.1, filter_small=False, filter_image=None, analyze_dilate=0, analyze_pad=0, analyze_thresh=1.2, verbose=True, remove_failed=True, **kwargs):
        """
        Recompute source parameters with a new catalog / segmentation image
        
        Parameters
        ----------
        data : array-like
            Intensity (science) data
        
        err : array-like
            Uncertainties
        
        seg : array-like (int)
            Segmentation image
        
        cat : `astropy.table.Table`
            Source catalog
        
        autoparams, flux_radii : list
            Parameters for `AUTO` / `KRON` attributes
        
        min_a : float
            Remove sources with semimajor axis smaller than this threshold
            `a_image < min_a`
            
        analyze_robust : int
            "Robust" analysis (see `golfir.catalog.analyze_image`)
        
        filter_small, filter_image : bool, array-like
            Measure parameters for `layer=0` sources on `data - filter_image`
            filtered  data.
        
        verbose : bool
           Print messages
           
        Returns
        -------
        new : `astropy.table.Table`
            Table with new source parameters (e.g, xmin, xmax, a_image, etc)
        
        seg : array-like
            New segmentation image cleaned of missing sources in `new` table
            
        Notes
        -----
        Output `new` table may not have same size as input `cat`, missing 
        sources where the analysis failed
            
        """
        if data_bkg is None:
            data_bkg = data
            
        if filter_small & ('layer' in cat.colnames):
            new = analyze_image(data - filter_image, 
                                err, seg, cat,
                                athresh=analyze_thresh, 
                                robust=analyze_robust,
                                allow_recenter=analyze_recenter,
                                prefix='', suffix='', grow=0, 
                                subtract_background=False, 
                                include_empty=False, 
                                pad=analyze_pad,
                                dilate=analyze_dilate,
                                make_image_cols=True)
        
            big = cat['layer'] > 0
            new2 = analyze_image(data_bkg, 
                                 err, seg, cat[big],
                                 athresh=analyze_thresh, 
                                 robust=analyze_robust,
                                 allow_recenter=analyze_recenter,
                                 prefix='', suffix='', grow=0, 
                                 subtract_background=False, 
                                 include_empty=False, 
                                 pad=analyze_pad,
                                 dilate=analyze_dilate,
                                 make_image_cols=True)
        
            for k in new2.colnames:
                new[k][big] = new2[k]
            
        else:
            new = analyze_image(data_bkg, 
                            err, seg, cat,
                            athresh=analyze_thresh, 
                            robust=analyze_robust,
                            allow_recenter=analyze_recenter,
                            prefix='', suffix='', grow=0, 
                            subtract_background=False,
                            include_empty=False, 
                            pad=analyze_pad,
                            dilate=analyze_dilate,
                            make_image_cols=True)
                    
        new.meta['FILTERSM'] = (filter_small,
                                'Layer 0 analysis on filtered image')
        
        sh = data.shape
        ok_x = (new['x_image'] > 1.0) & (new['x_image'] < sh[1])
        ok_x &= (new['y_image'] > 1.0) & (new['y_image'] < sh[0])
        ok_x &= np.isfinite(new['x_image']) & np.isfinite(new['y_image'])
        
        big_enough = new['a_image'] > min_a
        if verbose:
            print(f'Remove {(~ok_x).sum()} bad centroids, '
                  f'{(~big_enough).sum()} too small')

        valid = ok_x & big_enough
        #new = new[ok_x & big_enough]            
        
        new_ids = new['id']
        cat_ids = cat['id']
        #ix_new = np.array([np.where(cat_ids == n)[0][0] for n in new_ids])
        #cat['failed'] = (~valid)
        #cat['failed'][ix_new] = False
        
        for k in ['number','ra','dec','layer']:
            new[k] = cat[k]#[~cat['failed']]

        #cat['failed'] = ~ok
        
        #switch_segments(seg, cat[cat['failed']], cat['id'][~ok]*0)
        new['ix'] = np.arange(len(new))
        seg, clean = remove_missing_ids(seg, new[valid])
        new['failed'] = True
        new['failed'][clean['ix']] = False
        
        snew = get_seg_limits(seg)
        for k in snew.colnames:
            if k in ['id']:
                continue
            
            new[k] = snew[k][0]
            new[k][clean['ix']] = snew[k]
        
        # metadata
        for k in cat.meta:
            new.meta[k] = cat.meta[k]
        
        new.meta['FILTERSM'] = filter_small, 'Layer 0 analyzed on filtered image'
        ### ISO fluxes (flux within segments)
        iso_flux, iso_fluxerr, iso_area = prep.get_seg_iso_flux(data_bkg, seg, 
                                                 new[clean['ix']],
                                                 err=err, verbose=1)

        new['flux_iso'] = iso_flux[0]*0
        new['fluxerr_iso'] = iso_fluxerr[0]*0
        new['area_iso'] = iso_area[0]*0
        
        new['flux_iso'][clean['ix']] = iso_flux
        new['fluxerr_iso'][clean['ix']] = iso_fluxerr
        new['area_iso'][clean['ix']] = iso_area
        
        ### auto params
        # if filter_small & ('layer' in cat.colnames):
        #     auto = prep.compute_SEP_auto_params(data, data - filter_image, 
        #                                 err <= 0,
        #                                 pixel_scale=pixel_scale,
        #                                 err=err, segmap=seg,
        #                                 tab=new[clean['ix']],
        #                                 autoparams=autoparams, 
        #                                 flux_radii=flux_radii,
        #                                 subpix=0, verbose=True)
        # 
        #     big = new[clean['ix']]['layer'] > 0
        #     auto2 = prep.compute_SEP_auto_params(data, data_bkg, err <= 0,
        #                                 pixel_scale=pixel_scale,
        #                                 err=err, segmap=seg,
        #                                 tab=new[clean['ix']],
        #                                 autoparams=autoparams, 
        #                                 flux_radii=flux_radii,
        #                                 subpix=0, verbose=True)
        # 
        #     for k in auto.colnames:
        #         auto[k][big] = auto2[k][big]
        #     
        # else:
        
        ### Measure auto on bck-subtracted images, not filtered
        if True:
            auto = prep.compute_SEP_auto_params(data, data_bkg, err <= 0,
                                        pixel_scale=pixel_scale,
                                        err=err, segmap=seg,
                                        tab=new[clean['ix']],
                                        autoparams=autoparams, 
                                        flux_radii=flux_radii,
                                        subpix=0, verbose=True)
                
        for k in auto.colnames:
            new[k] = auto[k][0]*0
            new[k][clean['ix']] = auto[k]

        for k in auto.meta:
            new.meta[k] = auto.meta[k]
        
        new['mag_auto'] = ZP - 2.5*np.log10(new['flux_auto'])
        
        if remove_failed:
            new = new[~new['failed']]
            
        return new, seg


    def find_starjunk(self, seg, cat, flux_threshold=0.1, mag_to_size=(lambda mag: np.maximum(19-mag, 0)*0.7), star_mag_limit=19, stardef=[np.array([10, 16, 17, 18, 19, 20, 21, 23])-2.5, [10, 10, 5, 3.2, 2.1, 1.7, 1.7, 1.7]], **kwargs):
        """
        Remove fainter sources around stars identified based on `flux_radius`
        property in `new_cat` catalog
        
        Parameters
        ----------
        seg : array-like (int)
            Segmentation image
        
        cat : `astropy.table.Table`
            Source table with at least `mag_auto`, `flux_radius` and 
            positional columns
        
        flux_threshold : float
            Maximum flux ratio of junk to the associated star
            
        mag_to_size : func
            Function for converting `mag_auto` to a matching radius, in pixels
        
        star_mag_limit : float
            Faint limit on `mag_auto`
        
        stardef : [array, array]
            Selection line for stars in `mag_lim` and `flux_radius`
            
        Returns
        -------
        sfix, cfix : array-like, table
            Segmentation image and catalog cleaned of sources around 
            identified stars

        
        """
        
        #xstar = np.array([10, 16, 17, 18, 19, 20, 21, 23])-2.5
        #ystar = [10, 10, 5, 3.2, 2.1, 1.7, 1.7, 1.7]
        xstar, ystar = stardef
        ylim = np.interp(cat['mag_auto'], xstar, ystar)
        
        star = (cat['mag_auto'] < star_mag_limit) 
        star &= (cat['flux_radius'] < ylim)
        
        if star.sum() == 0:
            return seg, cat
            
        ix, dr = cat[star].match_to_catalog_sky(cat)
        near_star = (dr.value < mag_to_size(cat['mag_auto'][star][ix]))
        near_star &= (dr.value > 0)
        near_star &= cat['flux']/cat['flux'][star][ix] < flux_threshold

        cats = cat[star]
        cats['a_image'] = mag_to_size(cat['mag_auto'][star])/0.1
        cats['b_image'] = mag_to_size(cat['mag_auto'][star])/0.1
        
        cat['near_star'] = near_star
        
        msg = f'near_star: {near_star.sum()} sources'
        utils.log_comment(self.logfile, msg, verbose=True)
        
        prep.table_to_regions(cat[near_star], 
                              f'{self.root}-{self.filter}_near_star.reg', 
                              use_world=False, scale_major=3, 
                              use_ellipse=True, 
                              header='global color=red')

        prep.table_to_regions(cats, 
                              f'{self.root}-{self.filter}_star.reg',
                              use_world=False, 
                              scale_major=1, use_ellipse=True, 
                              header='global color=green')
        
        cat = cat[~near_star]
        
        sfix, cfix = remove_missing_ids(seg, cat)
        
        slim = get_seg_limits(sfix)
        for k in slim.colnames:
            if k in ['id']:
                continue

            cfix[k] = slim[k]
        
        return sfix, cfix
        #self.clean_cat = new


    def watershed_segmentation(self, cat, small_circle=6, expand_smallest=1, **kwargs):
        """
        Watershed segmentation of sources layered catalogs
        
        Parameters
        ----------
        cat : `astropy.table.Table`
            Source table with `layer` column
        
        seg : array-like (int)
            Original segmentation image
        
        Returns
        -------
        wcat : `astropy.table.Table`
            Associated table with new geometry parameters
        
        wseg : array-like (int)
            Watershed segmentation image
            
        Notes
        -----
        1. Thresholds set above in `pipeline`
        
        """
        #from skimage.segmentation import watershed
        try:
            from skimage.morphology import watershed
        except:
            from skimage.segmentation import watershed

        #from skimage.segmentation import expand_labels

        ##### Layer 0            
        xp = np.round(cat['x']).astype(int)
        yp = np.round(cat['y']).astype(int)
        
        fdata = (self.data - self.filtered[0])
        sn = fdata*np.sqrt(self.wdata)
        
        markers = np.zeros(self.sh, dtype=int)
        markers[yp, xp] = cat['id']*1
        
        markers, cat = remove_missing_ids(markers, cat, fill_value=0, 
                                          verbose=True,
                                          logfile=self.logfile)
        
        #m0 = expand_labels(markers, distance=expand_smallest)
        # little circle
        fp = circle_footprint(scale=small_circle, shrink=0.8)
        mx = markers + 1e8*(markers == 0)
        mi = nd.minimum_filter(mx, footprint=fp)
        m0 = mi * (mi < 1e7)
        
        seg = watershed(-fdata, markers, 
                        mask=(sn >= 1.))
        
        seg[m0 > 0] = m0[m0 > 0]
        
        slim = get_seg_limits(seg)
        
        switch_segments(seg, slim[cat['layer'] > 0])
        #seg = expand_labels(seg, distance=expand_smallest)
        
        ###########
        # Next layer        
        wseg = seg*1
        big = cat['layer'] > 0

        xp = np.round(cat['x']).astype(int)
        yp = np.round(cat['y']).astype(int)

        switch_segments(wseg, cat[big], cat['id'][big]*0)

        markers = np.zeros(self.sh, dtype=int)
        markers[yp[big], xp[big]] = cat['id'][big]
        
        wdata = (self.filtered[0] - self.filtered[1])
        
        sn = wdata*np.sqrt(self.wfiltered[1])
        w0 = watershed(-wdata, markers, 
                       mask=(sn >= self.watershed_thresholds[0]))

        wseg[wseg == 0] += w0[wseg == 0]
        wseg[m0 > 0] = m0[m0 > 0]
        
        # Objects lost from thresholding
        sub_lost = sn[yp[big], xp[big]] < self.watershed_thresholds[0]
        ix = np.where(big)[0]
        lost = cat['id'] < 0
        lost[ix[sub_lost]] = True
        #cat = cat[~lost]
        msg = f'Segmentation step 1 (threshold={self.watershed_thresholds[0]}): lost {sub_lost.sum()} sources'
        utils.log_comment(self.logfile, msg, verbose=True)
        
        wseg, cat = remove_missing_ids(wseg, cat, fill_value=0, verbose=True,
                           logfile=self.logfile)
        
        ##################
        # Next step
        xp = np.round(cat['x']).astype(int)
        yp = np.round(cat['y']).astype(int)

        big = cat['layer'] == 2
        newt = get_seg_limits(wseg)
        for k in newt.colnames:
            cat[k][big] = newt[k][big]*1

        switch_segments(wseg, cat[big])

        markers *= 0
        markers[yp[big], xp[big]] = cat['id'][big]
        if 0:
            wdata = (self.filtered[1])
            sn = wdata*np.sqrt(self.wfiltered[1])
        else:
            wdata = self.filtered[0]
            sn = wdata*np.sqrt(self.wdata)
            
        w0 = watershed(-wdata, markers, 
                       mask=(sn >= self.watershed_thresholds[1]))

        wseg[wseg == 0] += w0[wseg == 0]
        wseg[m0 > 0] = m0[m0 > 0]

        sub_lost = sn[yp[big], xp[big]] < self.watershed_thresholds[1]
        ix = np.where(big)[0]
        lost = cat['id'] < 0
        lost[ix[sub_lost]] = True
        #cat = cat[~lost]
        msg = f'Segmentation step 2 (threshold={self.watershed_thresholds[1]}): lost {sub_lost.sum()} sources'
        utils.log_comment(self.logfile, msg, verbose=True)
        
        wseg, cat = remove_missing_ids(wseg, cat, fill_value=0, verbose=True,
                           logfile=self.logfile)
        
        # Segment bbox
        newt = get_seg_limits(wseg)
        for k in newt.colnames:
            cat[k] = newt[k]
        
        #self.water_seg = sfill
        #self.water_cat = cat
        wcat = cat
        
        return wcat, wseg


    def final_outputs(self, **kwargs):
        """
        Write output files and re-run multiband photometry catalog based on 
        the new detection catalog
        """
        from grizli.pipeline import auto_script
        
        extra = []
        for i, layer in enumerate(self.water_cat['layer']):
            if layer == 0:
                extra.append('color=pink')
            elif layer == 1:
                extra.append('color=red')
            else:
                extra.append('color=orange')
                
        prep.table_to_regions(self.water_cat,
                              f'{self.root}-{self.filter}_final.reg',
                                      use_world=False, 
                                    scale_major=self.water_cat['kron_radius'],
                                      use_ellipse=True, 
                                      extra=extra)

        upper = utils.GTable()
        for c in self.water_cat.colnames:
            upper[c.upper()] = self.water_cat[c]

        for k in self.water_cat.meta:
            upper.meta[k] = self.water_cat.meta[k]

        upper['NUMBER'] = upper['ID']
        upper.remove_column('ID')
        upper['X_WORLD'] = upper['RA']
        upper['Y_WORLD'] = upper['DEC']

        source_xy = (upper['X_WORLD'], upper['Y_WORLD'], 
                     self.water_seg, upper['NUMBER'])
        
        # Apertures on detection image for aper corrections
        det_aper = prep.make_SEP_catalog(root=f'{self.root}-{self.filter}',
                  threshold=phot_args['threshold'],
                  rescale_weight=False,
                  err_scale=False,
                  get_background=True,
                  phot_apertures=phot_args['phot_apertures'],
                  save_to_fits=False, source_xy=source_xy,
                  bkg_mask=None,
                  bkg_params=phot_args['bkg_params'],
                  use_bkg_err=phot_args['use_bkg_err'])
        
        for k in det_aper.colnames:
            upper[k] = det_aper[k]
        
        for k in det_aper.meta:
            upper.meta[k] = det_aper.meta[k]
        
        # Write catalog    
        upper.write(f'{self.root}-{self.filter}.cat.fits', overwrite=True)
        
        # Write segmentation image
        pyfits.writeto(f'{self.root}-{self.filter}_seg.fits', 
                       data=self.water_seg, header=self.header, 
                       overwrite=True)

        phot_args['field_root'] = self.root
        phot_args['detection_filter'] = self.filter
        phot_args['run_detection'] = False
        
        # Full photometry
        auto_script.multiband_catalog(**phot_args)


    def remove_subcomponents(self, cat, scale_major=5, overlap_threshold=0.5, verbose=False, make_plot=False, parent_layers=[1,2], total_thresh=1.0, child_thresh=0.01, **kwargs):
        """
        Remove subcomponents
        """
        # frame = inspect.currentframe()
        # utils.log_function_arguments(self.logfile, frame, 
        #                              func='FilterDetection.remove_subcomponents')
        # 
        import sep
        
        pa = cat['layer'] < 0
        for l in parent_layers:
            pa |= cat['layer'] == l
                                
        pa = np.where(pa)[0]
        ch = cat['layer'] == 0
        
        points = np.array([cat['x'], cat['y']]).T
        
        if make_plot:
            fig, ax = plt.subplots(1,1,figsize=(10,10))
            ax.set_aspect(1)
            ax.grid()
            ax.scatter(*points[ch,:].T, color='k', 
                       alpha=0.03, marker='.', s=4)
            
        x, y = cat['x'], cat['y']
        a, b = cat['a_image']*scale_major, cat['b_image']*scale_major
        th = cat['theta_image']
        cat['p_id'] = -1
        cat['p_ix'] = -1
        
        for i in pa:
            ellp = patch_ellipse_from_catalog(x[i], y[i], 
                                               a[i], b[i], th[i], 
                                               color='orange', alpha=0.4)
                                               
            in_p = ellp.contains_points(points) & ch
            if in_p.sum() > 0:
                if verbose:
                    print(f'Parent {i}')
                if make_plot:
                    ax.add_patch(ellp)
                    
                pi = shapely_ellipse_from_catalog(x[i], y[i], 
                                                   a[i], b[i], th[i])
                for j in np.where(in_p)[0]:
                    if verbose:
                        print(f'  child {j}')
                    pj = shapely_ellipse_from_catalog(x[j], y[j], 
                                                       a[j], b[j], th[j])
                    
                    if pj.intersection(pi).area/pj.area > overlap_threshold:
                        if make_plot:
                            ax.plot(*pj.boundary.xy, color='r', alpha=0.5)
                            
                        cat['p_id'][j] = cat['id'][i]
                        cat['p_ix'][j] = i
        
        cat['is_parent'] = False
        cat['is_parent'][cat['p_ix']] = True
        
        show = cat['is_parent'] | (cat['p_ix'] > 0)
        
        _ephot, _, _ = sep.sum_ellipse(self.data, cat['x'], cat['y'],
                                cat['a_image']*scale_major/2, 
                                cat['b_image']*scale_major/2, 
                                cat['theta_image'], 1.0, subpix=0)
        
        child_frac = _ephot / _ephot[cat['p_ix']]
        extra = [0]*show.sum()
        isub = np.where(show)[0]
        remove = cat['id'] < -10
        
        for j, i in enumerate(isub):
            isp = cat['is_parent'][i]
            if isp:
                ch = cat['p_id'] == cat['id'][i]
                pflux = _ephot[i]
                cflux = _ephot[ch].sum()
                if (cflux/pflux < total_thresh) | (cat['layer'][i] == 2):
                    extra[j] = 'color=green'
                else:
                    extra[j] = 'color=red'
                    remove[i] = True
        
        for j, i in enumerate(isub):
            isp = cat['is_parent'][i]
            if not isp:
                if (child_frac[i] < child_thresh) & (not remove[cat['p_ix'][i]]):
                    extra[j] = 'color=pink'
                    remove[i] = True
                else:
                    extra[j] = 'color=cyan'
                                        
        prep.table_to_regions(cat[isub],
                              f'{self.root}-{self.filter}_parent.reg',
                                      use_world=False, 
                                      scale_major=scale_major/2,
                                      use_ellipse=True, 
                                      extra=extra)
        
        msg = f'Remove {remove.sum()} subcomponents'
        utils.log_comment(self.logfile, msg, verbose=True)
        
        return cat[~remove]


    def pipeline(self, watershed_thresholds=[0.3, 1.8], run_remove_subcomponents=False, clean_junk=True, **kwargs):
        """
        Run the whole thing. 
        """
        frame = inspect.currentframe()
        utils.log_function_arguments(self.logfile, frame, 
                                    func='FilterDetection.pipeline')
        
        import sep
        
        self.watershed_thresholds = watershed_thresholds
        
        # Median filter
        utils.log_comment(self.logfile, 'Median filtering', 
                          verbose=True, show_date=True)
        self.smooth_scales(**kwargs)
        
        # Run source detection
        utils.log_comment(self.logfile, 'Source detection', 
                          verbose=True, show_date=True)
        self.run_detection(**kwargs)
        
        # Combine layer catalogs
        utils.log_comment(self.logfile, 'Combine catalogs', 
                          verbose=True, show_date=True)                          
        self.det_cat = self.combine_catalogs(**kwargs)
        
        extra = []
        for i, layer in enumerate(self.det_cat['layer']):
            if layer == 0:
                extra.append('color=pink')
            elif layer == 1:
                extra.append('color=red')
            else:
                extra.append('color=orange')
                
        prep.table_to_regions(self.det_cat,
                              f'{self.root}-{self.filter}_detect.reg',
                                      use_world=False, scale_major=3,
                                      use_ellipse=True, 
                                      extra=extra)
        
        _ = self.watershed_segmentation(self.det_cat, **kwargs)
        self.init_cat, self.init_seg = _
        
        # Combine segments across layers
        #utils.log_comment(self.logfile, 'Combine segments', 
        #                  verbose=True, show_date=True)
        #self.combine_segments(**kwargs)
        
        # Analyze new catalog
        to_ujy = 1./self.cat0.meta['uJy2dn'][0]
        data = self.data*to_ujy
        err = 1/np.sqrt(self.wdata)*to_ujy
        err[self.wdata <= 0] = 0
        self.init_seg, self.init_cat = remove_missing_ids(self.init_seg, 
                                                        self.init_cat)
        
        xlim = get_seg_limits(self.init_seg)
        for k in xlim.colnames:
            self.init_cat[k] = xlim[k]
            
        seg = self.init_seg*1
        
        # Photometry background
        bw = phot_args['bkg_params']['bw']
        fw = phot_args['bkg_params']['fw']

        bkg = sep.Background(data, mask=(err <= 0), bw=bw, bh=bw,
                             fw=fw, fh=fw)
        data_bkg = data - bkg.back()
        
        self.new_cat, seg = self.reanalyze_image(data, err, seg, 
                                                self.init_cat,
                                                data_bkg=data_bkg,
                                         filter_image=self.filtered[1]*to_ujy,
                                                ZP=23.9,
                                                pixel_scale=self.pixel_scale,
                                                **kwargs)

        extra = []
        for i, layer in enumerate(self.new_cat['layer']):
            if layer == 0:
                extra.append('color=pink')
            elif layer == 1:
                extra.append('color=red')
            else:
                extra.append('color=orange')
                
        prep.table_to_regions(self.new_cat,
                              f'{self.root}-{self.filter}_new.reg',
                                      use_world=False, scale_major=3,
                                      use_ellipse=True, 
                                      extra=extra)

        # Remove starjunk
        if clean_junk:
            utils.log_comment(self.logfile, 'Remove starjunk', 
                          verbose=True, show_date=True)
                          
            self.clean_seg, self.clean_cat = self.find_starjunk(seg, 
                                                            self.new_cat, 
                                                            **kwargs)
        else:
            self.clean_seg, self.clean_cat = seg, self.new_cat
            
        # Watershed segmentation
        utils.log_comment(self.logfile, 'Watershed segmentation', 
                          verbose=True, show_date=True)
        
        _ = self.watershed_segmentation(self.clean_cat, **kwargs)
        self.water_cat, self.water_seg = _
        
        self.water_cat, self.water_seg = self.reanalyze_image(data, err, 
                                              self.water_seg, self.water_cat, 
                                              data_bkg=data_bkg,
                                         filter_image=self.filtered[1]*to_ujy,
                                              ZP=23.9,
                                              pixel_scale=self.pixel_scale,
                                              **kwargs)
        
        # remove small sources near the center of layer=2 
        layer2 = self.water_cat['layer'] == 2
        keep = np.isfinite(self.water_cat['id'])
        
        for i in np.where(layer2)[0]:
            dx = self.water_cat['x'] - self.water_cat['x'][i]
            dy = self.water_cat['y'] - self.water_cat['y'][i]
            dr = np.sqrt(dx**2+dy**2)
            near = (dr > 0) & (dr < 8)
            keep[near] = False
        
        msg = f'Trim {(~keep).sum()} sources close to center of large objects'
        utils.log_comment(self.logfile, msg, verbose=True)
        
        extra = []
        for i, layer in enumerate(self.water_cat['layer']):
            if layer == 0:
                extra.append('color=pink')
            elif layer == 1:
                extra.append('color=red')
            else:
                extra.append('color=orange')
                
        prep.table_to_regions(self.water_cat[~keep],
                              f'{self.root}-{self.filter}_xtrim.reg',
                                      use_world=False, scale_major=3,
                                      use_ellipse=True, 
                              extra=[extra[i] for i in np.where(~keep)[0]])
        
        prep.table_to_regions(self.water_cat[keep],
                              f'{self.root}-{self.filter}_trim.reg',
                                      use_world=False, scale_major=3,
                                      use_ellipse=True, 
                                      extra=[extra[i] for i in np.where(keep)[0]])
        
        self.water_cat = self.water_cat[keep]
        
        # Trim sources that grow significantly
        ilist = self.det_cat['new_id'].tolist()
        ixc = np.array([ilist.index(i) for i in self.water_cat['id']])
        bad = self.water_cat['a_image']/self.det_cat['a_image'][ixc] > 5
        bad &= self.water_cat['layer'] < 2
        msg = f'Trim {(bad).sum()} sources that grow between det->water'
        utils.log_comment(self.logfile, msg, verbose=True)
        
        prep.table_to_regions(self.water_cat[bad],
                              f'{self.root}-{self.filter}_grow.reg',
                                      use_world=False, scale_major=3,
                                      use_ellipse=True)
        
        self.water_cat = self.water_cat[~bad]
        
        # Remove small subcomponents
        if run_remove_subcomponents:
            self.water_cat = self.remove_subcomponents(self.water_cat, 
                                                       **kwargs)
        
        _ = self.watershed_segmentation(self.water_cat, **kwargs)
        self.water_cat, self.water_seg = _
        
        self.water_cat, self.water_seg = self.reanalyze_image(data, err, 
                                              self.water_seg, self.water_cat, 
                                              data_bkg=data_bkg,
                                         filter_image=self.filtered[1]*to_ujy,
                                              ZP=23.9,
                                              pixel_scale=self.pixel_scale,
                                              **kwargs)
        
        _ = remove_missing_ids(self.water_seg, self.water_cat, 
                               fill_value=0, verbose=True)
        
        self.water_seg, self.water_cat = _
        
        # Write output products 
        self.final_outputs(**kwargs)
        