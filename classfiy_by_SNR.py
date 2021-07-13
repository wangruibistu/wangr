import os
import astropy.io.fits as pf
import numpy as np
import pandas as pd
import fitsname_to_specid as fs
import get_param_from_lamost_sql as gp
import parameters.GSN.mcmc_test_phoenix_v2 as mc_apo


def classfy(path):
    filelist = os.listdir(path)
    df = pd.DataFrame([])
    fitsname = []
    specid = []
    snrs = []
    teffs = []
    loggs = []
    fehs = []
    rvs = []
    for fits in filelist:
        fitsname.append(fits)
        spid = fs.getSpecidFromName(fits)
        specid.append(spid)
        hdu = pf.open(path + fits)
        sng = hdu[0].header['SN_G']
        snrs.append(sng)
        teff, teff_err, logg, logg_err, feh, feh_err, rv, rv_err = gp.get_parameters(spid)
        teffs.append(teff)
        loggs.append(logg)
        fehs.append(feh)
        rvs.append(rv)  # snr = hdu[0].header['snr']

    df['fits'] = fitsname
    df['specid'] = specid
    df['snr'] = snrs
    df['teff'] = teffs
    df['logg'] = loggs
    df['feh'] = fehs
    df['rvs'] = rvs

    for j in 10 * np.arange(1, 10):
        # print j
        df_tmp = df[(j < df.snr) & (df.snr <= j + 10)]
        n_all = len(df_tmp)
        params = df_tmp.loc[:, ['specid', 'teff', 'logg', 'feh', 'rvs']]
        fitslist = df_tmp.fits.values
        # print n_all, fitslist, path, params, j
        mc_apo.multi(n_all, fitslist, path, params, j)
    df_tmp = df[100 < df.snr]
    n_all = len(df_tmp)
    params = df_tmp.loc[:, ['specid', 'teff', 'logg', 'feh', 'rvs']].values
    fitslist = df_tmp.fits.values
    mc_apo.multi(n_all, fitslist, path, params, 100)


if __name__ == '__main__':
    classfy('/home/wangr/data/LAMOST_cross/lamost_gcs_new/')
