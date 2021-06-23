
def get_eso():
    import astroquery.eso
    eso = astroquery.eso.Eso()
    
    # Configure 'username' in ~/.astropy/config
    eso.login() 
    return eso
    
def full_query(filters=['Ks'], min_nexp=5, eso=None):
    
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
    
    kwargs['columns'] = ['tpl_nexp', 'tpl_expno', 'det_ndit', 'det_dit', 'det_ncorrs_name', 'obs_tmplno', 'det_ncorrs_name', 'prog_type', 'pi_coi']
    
    res = eso.query_instrument('hawki', pi_coi_name='PI_only', **kwargs)
    res['PI'] = [p.split('/')[0].strip() for p in res['PI/CoI']]
    return eso, kwargs, res
    
def full_hawki_query(rd=None, query_result=None, eso=None):
    """
    Query all HAWKI observations....
    """ 
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    
    from shapely.geometry import Polygon, Point
    from descartes import PolygonPatch
    from shapely import affinity 
    
    from grizli import utils
    from mastquery import query, overlaps
    
    if eso is None:
        eso = get_eso()
        
    if query_result is None:
        _, kwargs, res = full_query(eso=eso)
    else:
        kwargs, res = query_result
        
    # surveys = 092.A-0472 
    
    # CHArGE fields
    from grizli.aws import db
    import astropy.units as u
    from astropy.coordinates import SkyCoord 
    
    engine = db.get_db_engine()
    if rd is None:
        ch = db.from_sql("SELECT field_root, field_ra as ra, field_dec as dec, log FROM charge_fields where log LIKE '%%Finish%%'", engine)
    else:
        ra, dec = rd
        ch = utils.GTable()
        ch['ra'] = [ra]
        ch['dec'] = [dec]

        ch['field_root'] = [utils.radec_to_targname( ra=ra, dec=dec, round_arcsec=(4, 60), precision=2, targstr='j{rah}{ram}{ras}{sign}{ded}{dem}', header=None, )]
        
    idx, dr = ch.match_to_catalog_sky(res)
    
    has_hawki = dr < 10*u.arcmin
    
    import scipy.spatial
    ch_rd = SkyCoord(ch['ra'], ch['dec'], unit='deg')
    ch_xyz = ch_rd.cartesian.get_xyz().value
    ctree = scipy.spatial.cKDTree(ch_xyz.T)
    
    hawki_rd = SkyCoord(res['RA'], res['DEC'], unit='deg')
    hawki_xyz = hawki_rd.cartesian.get_xyz().value
    htree = scipy.spatial.cKDTree(hawki_xyz.T)
    
    r = 30./60/360.*2
    
    tr = ctree.query_ball_tree(htree, r)
    n_hawki = np.array([len(t) for t in tr])
    
    # Figures    
    idx = np.where(n_hawki > 0)[0]
    
    xsize = 5
    px, py = 0.45, 0.2
    
    for i in idx:
        field = ch['field_root'][i]
        print(i, field)
        if os.path.exists(f'{field}_hawki.png'):
            continue
            
        field = ch['field_root'][i]

        #tab = utils.read_catalog(f'../FieldsSummary/{field}_footprint.fits')
        if os.path.exists(f'{field}_footprint.fits'):
            tab = utils.read_catalog(f'{field}_footprint.fits')
            meta = tab.meta
        
            xr = (meta['XMIN'], meta['XMAX'])
            yr = (meta['YMIN'], meta['YMAX'])
            ra, dec = meta['BOXRA'], meta['BOXDEC']
        
            cosd = np.cos(dec/180*np.pi)
            dx = (xr[1]-xr[0])*cosd*60
            dy = (yr[1]-yr[0])*60

            box_width = np.maximum(dx, dy)
            #query_size = np.maximum(min_size, box_width/2)/60.
        
            p_hst = None
            p_ir = None
        
            for j, fph in enumerate(tab['footprint']):
                ps, is_bad, poly = query.instrument_polygon(tab[j])
                if not hasattr(ps, '__len__'):
                    ps = [ps]
                
                for p in ps:
                    p_j = Polygon(p).buffer(0.001)
                    if p_hst is None:
                        p_hst = p_j
                    else:
                        p_hst = p_hst.union(p_j)
                
                    if tab['instrument_name'][j] == 'WFC3/IR':
                        if p_ir is None:
                            p_ir = p_j
                        else:
                            p_ir = p_ir.union(p_j)
        else:
            cosd = np.cos(dec/180*np.pi)
            p_hst = None
            p_ir = None
            
        ##############################            
        fig = plt.figure(figsize=[6,6])
        
        ax = fig.add_subplot(111)
        ax.scatter(ra, dec, zorder=1000, marker='+', color='k')
        
        # HAWKI
        h_p = None
        for j in tr[i]:
            p = Point(res['RA'][j], res['DEC'][j]).buffer(4.1/60)
            p = affinity.scale(p, xfact=1./cosd)
            
            # ax.add_patch(PolygonPatch(p, color='r', alpha=0.1))
            x, y = p.boundary.xy
            ax.plot(x, y, color=utils.MPL_COLORS['r'], alpha=0.05)
            
            if h_p is None:
                h_p = p
            else:
                h_p = h_p.union(p)
        
        # If overlap between hawki and HST, query all exposures
        if p_hst is not None:
            hawki_overlap = h_p.intersection(p_hst)
            hawki_un = h_p.union(p_hst)
                        
            if not hasattr(p_hst, '__len__'):
                p_hst = [p_hst]
        
            if not hasattr(h_p, '__len__'):
                h_p = [h_p]
                                                            
            for p in p_hst:
                #ax.add_patch(PolygonPatch(p, color='k', alpha=0.2))
                if not hasattr(p.boundary, '__len__'):
                    bs = [p.boundary]
                else:
                    bs = p.boundary
            
                for b in bs:
                    x, y = b.xy
                    ax.plot(x, y, color=utils.MPL_COLORS['gray'], alpha=0.3)
        else:
            hawki_overlap = h_p
            if not hasattr(h_p, '__len__'):
                h_p = [h_p]
            
        if p_ir is not None:
            if not hasattr(p_ir, '__len__'):
                p_ir = [p_ir]
        
            for p in p_ir:
                ax.add_patch(PolygonPatch(p, color=utils.MPL_COLORS['gray'], alpha=0.2))
                x, y = p.boundary.xy
                ax.plot(x, y, color=utils.MPL_COLORS['gray'], alpha=0.3)
                                        
        for p in h_p:
            ax.add_patch(PolygonPatch(p, color=utils.MPL_COLORS['r'], alpha=0.2))

            
        targets = ['{0}  {1}'.format(res['ProgId'][j], res['Object'][j]) for j in tr[i]]
        for j, targ in enumerate(np.unique(targets)):
            ixj = np.where(np.array(targets) == targ)[0]
            expt = res['DET NDIT']*res['DET DIT']*res['TPL NEXP']
            
            ax.text(0.02, 0.98-j*0.03, '{0} {1:.1f}'.format(targ, expt[tr[i]][ixj].sum()/3600.), ha='left', va='top', transform=ax.transAxes, fontsize=7)
        
        ax.set_aspect(1./cosd)
        ax.set_title(field)
        ax.grid()
        
        #xsize = 4
        
        dx = np.diff(ax.get_xlim())[0]*cosd*60
        dy = np.diff(ax.get_ylim())[0]*60
        
        fig.set_size_inches(xsize*np.clip(dx/dy, 0.2, 5)+px, xsize+py)
        ax.set_xlim(ax.get_xlim()[::-1])
        overlaps.draw_axis_labels(ax=ax, nlabel=3)
        
        fig.tight_layout(pad=0.5)
        fig.savefig(f'{field}_hawki.png', dpi=120)
        plt.close('all')
        
        if (hawki_overlap.area > 0.0) & (not os.path.exists(f'{field}_hawki.fits')):
            
            kws = {}
            for k in kwargs:
                kws[k] = kwargs[k].copy()
            
            kws['column_filters'].pop('tpl_nexp')
            kws['column_filters'].pop('tpl_expno')
            
            _res = eso.query_instrument('hawki', pi_coi_name='PI_only',
                                        coord1=ra,
                                        coord2=dec, 
                                        box='00 30 00', 
                                        **kws)
                                        
            if len(_res) > 0:
                print('{0} datasets'.format(len(_res)))
                _res['PI'] = [p.split('/')[0].strip() for p in _res['PI/CoI']]
                _res.write(f'{field}_hawki.fits', overwrite=True)
        