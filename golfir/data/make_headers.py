"""
Get SIP headers for CBCD files
"""
import os
import astropy.io.fits as pyfits
import astropy.wcs as pywcs

def go():
    from grizli import utils
    
    os.system('aws s3 sync s3://grizli-v1/IRAC/AORS/r48106752/ch1/bcd/ ./ --exclude "*" --include "*752_0001*"')
    os.system('aws s3 sync s3://grizli-v1/IRAC/AORS/r48106752/ch2/bcd/ ./ --exclude "*" --include "*752_0001*"')
    
    for ch in [1,2]:
        im = pyfits.open(f'SPITZER_I{ch}_48106752_0001_0000_2_xbcd.fits.gz')
        wcs = pywcs.WCS(im['WCS'].header, relax=True)
        h = utils.to_header(wcs)
        h.totextfile(f'bcd_ch{ch}.header', overwrite=True)
    #
    #r25412864-ch4.log.fits
    os.system('aws s3 sync s3://grizli-v1/IRAC/AORS/r25412864/ch3/bcd/ ./ --exclude "*" --include "*864_0001*"')
    os.system('aws s3 sync s3://grizli-v1/IRAC/AORS/r25412864/ch4/bcd/ ./ --exclude "*" --include "*864_0001*"')
    
    for ch in [3,4]:
        im = pyfits.open(f'SPITZER_I{ch}_25412864_0001_0000_5_xbcd.fits.gz')
        wcs = pywcs.WCS(im['WCS'].header, relax=True)
        h = utils.to_header(wcs)
        h.totextfile(f'bcd_ch{ch}.header', overwrite=True)