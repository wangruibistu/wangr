import numpy as np
import keras.models as km
import os
import sys

sys.path.append('./..')
sys.path.extend(['/home/wangrui/code/python'])
import download_lamostfits as dl
import astropy.io.fits as pf
import random
import pandas as pd
from scipy.optimize import *
from scipy import exp, polyval, interpolate
from sklearn.externals import joblib
from multiprocessing import Process, Pool
import scipy.stats as ss

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

logwave_model = np.arange(3.5798, 3.9445, 0.0001)


def log_flags(wave):
    flag = np.ones_like(wave, dtype=bool)
    for i in range(len(wave)):
        if np.log10(5700) <= wave[i] <= np.log10(5900):
            flag[i] = False
        elif np.log10(6270) <= wave[i] <= np.log10(6320):
            flag[i] = False
        elif i == len(wave) - 1:
            flag[i] = False
        else:
            flag[i] = True
    return flag


logwave_model = logwave_model[log_flags(logwave_model)]
wave_model = 10 ** logwave_model
blue_tmp = np.where(wave_model < 5800)
red_tmp = np.where(wave_model > 5800)


def p_param(gmm_model, logteff, logg, feh):
    teff_norm = (10 ** logteff - 5562.3065673) / 750.63110289
    logg_norm = (logg - 3.98591330105) / 0.705530989523
    feh_norm = (feh + 0.236687317161) / 0.340038697846
    p = gmm_model.score_samples(np.array([teff_norm, logg_norm, feh_norm]).reshape(1, 3))
    return np.exp(p)


def fit_p3(wl, syn, obs):
    p0 = [0, 0, 0, 0, 1]
    func = lambda p, wl: polyval(p, wl) * syn
    err_func = lambda p, wl, obs: func(p, wl) - obs
    p, cov, info, mesg, success = leastsq(err_func, p0, args=(wl.copy(), obs.copy()), full_output=1)
    return p


def fit_p2(wl, syn, obs):
    p0 = [0, 0, 0, 1]
    func = lambda p, wl: polyval(p, wl) * syn
    err_func = lambda p, wl, obs: func(p, wl) - obs
    p, cov, info, mesg, success = leastsq(err_func, p0, args=(wl.copy(), obs.copy()), full_output=1)
    return p


def dist(kf, of):
    p_blue = fit_p3(wave_model[blue_tmp], kf[blue_tmp], of[blue_tmp])
    cont_blue = polyval(p_blue, wave_model[blue_tmp]) * kf[blue_tmp]
    d_blue = sum(((of[blue_tmp] - cont_blue) / 0.02) ** 2) / 1761

    p_red = fit_p2(wave_model[red_tmp], kf[red_tmp], of[red_tmp])
    cont_red = polyval(p_red, wave_model[red_tmp]) * kf[red_tmp]
    d_red = sum(((of[red_tmp] - cont_red) / 0.02) ** 2) / 1700

    d = d_blue + d_red
    cont = np.append(cont_blue, cont_red)

    return d, cont


def mcmc_z(gsn_model, gmm_model, teff, logg, feh, flux_obs):
    pso_spectrum0 = gsn_model.predict(np.array([[teff, logg, feh, 0]])).reshape(3461, )
    d_old, cont = dist(pso_spectrum0, flux_obs)
    teff_mcmc = np.array([teff])
    logg_mcmc = np.array([logg])
    feh_mcmc = np.array([feh])
    alpha_fe_mcmc = np.array([0])
    dis = np.array([d_old])
    param = np.array([teff, logg, feh, 0])
    num = 0
    p_old = exp(-0.5 * d_old) * p_param(gmm_model, teff_mcmc, logg_mcmc, feh_mcmc)
    while num < 1000:
        num += 1
        try:
            param_new = param + np.array(
                [random.uniform(-0.01, 0.01), random.uniform(-0.01, 0.01), random.uniform(-0.01, 0.01),
                    random.uniform(-0.01, 0.01)])
            pso_spectrum_predict = gsn_model.predict(param_new.reshape(1, 4)).reshape(3461, )

            d_new, c = dist(pso_spectrum_predict, flux_obs)  # , pixerrmean, pixerrstd
            p_new = exp(-0.5 * d_new) * p_param(gmm_model, param_new[0], param_new[1], param_new[2])
            alpha = min(p_new / p_old, 1)
            u = random.uniform(0, 1)
            if alpha > u:
                # print 'accept!'
                teff_mcmc = np.append(teff_mcmc, param_new[0])
                logg_mcmc = np.append(logg_mcmc, param_new[1])
                feh_mcmc = np.append(feh_mcmc, param_new[2])
                alpha_fe_mcmc = np.append(alpha_fe_mcmc, param_new[3])
                dis = np.append(dis, d_new)
                p_old = p_new
                param = param_new
        except RuntimeError:
            pass

    tmp = np.argmin(dis)
    pa = np.array([teff_mcmc[tmp], logg_mcmc[tmp], feh_mcmc[tmp], alpha_fe_mcmc[tmp]])
    pso_spectrum_predict_final = gsn_model.predict(pa.reshape(1, 4)).reshape(3461, )
    d, pred_ploy = dist(pso_spectrum_predict_final, flux_obs)
    # print 'min distance:', d
    output = str(10 ** teff_mcmc[tmp]) + ',' + str(logg_mcmc[tmp]) + ',' + str(feh_mcmc[tmp]) + ',' + str(
        alpha_fe_mcmc[tmp]) + ',' + str(min(dis)) + ',' + str(10 ** np.mean(teff_mcmc)) + ',' + str(
        np.mean(logg_mcmc)) + ',' + str(np.mean(feh_mcmc)) + ',' + str(np.mean(alpha_fe_mcmc)) + '\n'
    errs = flux_obs - pred_ploy
    return output, errs.reshape(1, 3461)


def lamost_fitsinfo(fitsfile):
    hdu = pf.open(fitsfile)
    z_1D = hdu[0].header['1D_Z']
    beginwave = hdu[0].header['CRVAL1']
    step = hdu[0].header['CD1_1']
    row = hdu[0].header['NAXIS1']
    logwave_obs = beginwave + step * np.arange(row)
    wl = 10 ** logwave_obs
    wave_rest = wl / (1 + z_1D)
    flux = np.array(hdu[0].data[0])
    flux_obs = flux / np.mean(flux)
    andmask = np.array(hdu[0].data[3])
    flux_obs_z = interpolate.UnivariateSpline((wave_rest)[np.where(andmask == 0)], flux_obs[np.where(andmask == 0)],
        s=0)(wave_model)
    return flux_obs_z


def run(n1, n2, fitspath, params, n_snr):
    output = []
    errs = np.arange(3461)
    gmm_model = joblib.load('/home/wangr/data/wr_data/LAMOST/gmm_model.m')
    gann_model = km.load_model('/home/wangr/code/python/parameters/phoenix/gsn_phoenix_n3461.model')
    for i in range(n1, n2):
        specid = params.specid.values[i]
        fitsname = dl.getSpecPath(specid).split('/')[-1]
        fitsfile = fitspath + fitsname
        flux_obs_z = lamost_fitsinfo(fitsfile)
        teff_lasp, logg_lasp, feh_lasp = params.iloc[:, -3:].values[i:i + 1][0]
        outcome, errs_flux = mcmc_z(gann_model, gmm_model, np.log10(teff_lasp), logg_lasp, feh_lasp, flux_obs_z)
        output.append(specid + ',' + str(teff_lasp) + ',' + str(logg_lasp) + ',' + str(feh_lasp) + ',' + outcome)
        errs = np.vstack((errs, errs_flux))
    errs = np.delete(errs, 0, 0)
    with open('/home/wangr/data/wr_data/LAMOST/dr5/pix_errs/' + 'phoenix_params_snr%s_%s_%s_%s.csv' % (
    str(n_snr), str(n_snr + 10), str(n1), str(n2)), 'w') as save:
        save.writelines(output)
    np.savetxt('/home/wangr/data/wr_data/LAMOST/dr5/pix_errs/' + 'phoenix_errs_snr%s_%s_%s_%s.csv' % (
    str(n_snr), str(n_snr + 10), str(n1), str(n2)), errs)


def multi(path, params, n_snr):
    fitslist = os.listdir(path)
    n_all = len(fitslist)
    n = n_all / 500
    # print 'n_all:', n
    tasks = []
    if n > 0:
        ns1 = np.arange(n + 1) * 500
        ns2 = np.append(ns1[1:], n_all)
        for k in range(n + 1):
            n1 = ns1[k]
            n2 = ns2[k]
            # print n1, n2
            tasks.append(Process(target=run, args=(n1, n2, path, params, n_snr)))
        for task in tasks:
            task.start()
        for task in tasks:
            task.join()
    if n == 0:
        tasks.append(Process(target=run, args=(0, n_all, path, params, n_snr)))
        for task in tasks:
            task.start()
        for task in tasks:
            task.join()


def data_clean(df):
    return df[(df > (df.mean() - 1 * df.std())) & (df < (df.mean() + 1 * df.std()))]


def mustd(errsfile, n_snr):
    path = '/home/wangr/data/wr_data/LAMOST/dr5/pix_errs/new/'
    df = pd.read_csv(path + errsfile, header=None, sep=' ')
    n_col = len(df.columns)
    pix_errs = np.array([])
    pix_std = np.array([])
    for i in range(n_col):
        # if i != 0:
        data = df.iloc[:, i:i + 1]
        data = data_clean(data).dropna()
        p = ss.norm.fit(data.values)
        pix_errs = np.append(pix_errs, p[0])
        pix_std = np.append(pix_std, p[1])
    # print len(pix_std)
    # print len(pix_errs)

    np.savetxt(
        '/home/wangr/data/wr_data/LAMOST/dr5/pix_errs/mustd/errmean_snr%s_%s.csv' % (str(n_snr), str(n_snr + 10)),
        pix_errs, delimiter=',')
    np.savetxt('/home/wangr/data/wr_data/LAMOST/dr5/pix_errs/mustd/errstd_snr%s_%s.csv' % (str(n_snr), str(n_snr + 10)),
        pix_std, delimiter=',')


def analysis(errfilepath, paramfilepath):
    # print errfilepath
    # print paramfilepath
    errsfile = pd.read_csv(errfilepath, header=None, sep=' ')
    param = pd.read_csv(paramfilepath, header=None)
    n_col = len(errsfile)
    errs=[]
    for i in range(n_col):
        # if i != 0:
        df = errsfile.iloc[i:i + 1,:].T
        df.columns=['w']
        df_new = data_clean(df.w)
        a=np.array([e**2 for e in df_new.values])
        errs.append(a.sum())

    param['ksq']=errs

    return param


if __name__ == '__main__':
    for i in 10 * np.arange(1, 10):
        parampath='/home/wangr/data/wr_data/LAMOST/dr5/pix_errs/phoenix_params_snr%.f_%.f.csv' % (i, i + 10)
        errspath = '/home/wangr/data/wr_data/LAMOST/dr5/pix_errs/phoenix_errs_snr%.f_%.f.csv' % (i, i + 10)
        param=analysis(errspath, parampath)
        # print param.ksq.describe()
        # params = pd.read_csv('/home/wangr/data/wr_data/LAMOST/dr5_param_snr%.f_%.f.csv' % (i, i + 10), header=None)
        # params.columns = ['specid', 'teff', 'logg', 'feh']
        # n_snr = i
        # multi(path, params, n_snr)
    # path = '/home/wangr/data/wr_data/LAMOST/dr5_param_snr100_/'
    # params = pd.read_csv('/home/wangr/data/wr_data/LAMOST/dr5_param_snr100_.csv', header=None)
    # params.columns = ['specid', 'teff', 'logg', 'feh']
    # n_snr = 100
    # multi(path, params, n_snr)
    # for n_snr in np.arange(1, 10) * 10:
    #     mustd('phoenix_errs_snr%s_%s.csv' % (str(n_snr), str(n_snr + 10)), n_snr)
    #  change_fmt('phoenix_errs_snr%s_%s.csv' % (str(n_snr), str(n_snr + 10)))
    #  change_fmt('phoenix_errs_snr100_.csv')
    #  mustd('phoenix_errs_snr100_.csv',100)
