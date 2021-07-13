import numpy as np
import matplotlib.pyplot as plt
import keras.models as km
import os
import sys
import seaborn as sns

sys.path.append('./..')
import fitsname_to_specid as f_to_s
import astropy.io.fits as pf
import random
import pandas as pd
from scipy.optimize import *
from scipy import exp, polyval, interpolate, signal
import scipy.stats as ss
from matplotlib import gridspec
from PyAstronomy import pyasl
from multiprocessing import Process, Pool

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def log_flags(wave):
    weights = np.ones_like(wave)
    flag = np.ones_like(wave, dtype=bool)
    for i in range(len(wave)):
        if np.log10(5700) <= wave[i] <= np.log10(5900):
            weights[i] = 0
            flag[i] = False
        elif np.log10(6270) <= wave[i] <= np.log10(6320):
            weights[i] = 0
            flag[i] = False
        elif np.log10(6549) <= wave[i] <= np.log10(6589):
            weights[i] = 2
        elif i == len(wave) - 1:
            weights[i] = 0
            flag[i] = False
        else:
            weights[i] = 1
            flag[i] = True
    return flag, weights


def gaus(x, a, mu, sigma, b):
    return a * exp(-(x - mu) ** 2 / (2 * sigma ** 2)) + b


def gaussfit(x, y):
    mean = sum(x * y) / sum(y)
    sigma = np.sqrt(sum(y * (x - mean) ** 2) / sum(y))
    a = -max(y) + min(y)
    b = max(y)
    popt, pcov = curve_fit(gaus, x, y, p0=[a, mean, sigma, b], maxfev=10000)
    return popt


def cross_correlation(x, y):
    return sum(x * y) / (np.sqrt(np.sum(x ** 2)) * np.sqrt(np.sum(y ** 2)))


def move_rv(obs_wave, obs_flux, obs_andmask, model_wave, model_flux):
    k_win = 5
    model_wave = np.arange(3.5798, 3.9445, 0.0001)
    move_step = 0.0001
    zs = np.array([])
    chis = np.array([])
    dists = np.array([])
    for i in range(-10, 11):
        z = i * move_step
        model_wave_z = model_wave + z

        tmp = np.where((obs_wave > np.min(model_wave_z) - 0.00001) & (obs_wave < np.max(model_wave_z) + 0.00001))[0]

        andmask_tmp = obs_andmask[tmp]
        wave = model_wave_z[np.where(andmask_tmp == 0)]
        obs_flux_tmp = obs_flux[tmp][np.where(andmask_tmp == 0)]
        model_flux_tmp = model_flux[np.where(andmask_tmp == 0)]

        p = fit_p3(wave, model_flux_tmp, obs_flux_tmp)
        cont_flux = polyval(p, wave) * model_flux_tmp

        corr = cross_correlation(obs_flux_tmp, cont_flux)
        # dist_tmp = sum((obs_flux_tmp - cont_flux) ** 2)
        zs = np.append(zs, z)
        chis = np.append(chis, corr)  # dists = np.append(dists, dist_tmp)

    max_location = np.argmax(chis)

    i_min = max(0, max_location - k_win)
    i_max = min(len(obs_flux_tmp) - 1, max_location + k_win)
    gauss_x = zs[i_min:i_max + 1]
    gauss_y = chis[i_min:i_max + 1]

    gaussfit_p = gaussfit(gauss_x, gauss_y)
    ksquare_bestfit = gaus(gaussfit_p[1], *gaussfit_p)

    return gaussfit_p[1], gaussfit_p[2], max(chis)


def fit_p3(wl, syn, obs):
    p0 = [0, 0, 0, 1]
    func = lambda p, wl: polyval(p, wl) * syn
    err_func = lambda p, wl, obs: func(p, wl) - obs
    p, cov, info, mesg, success = leastsq(err_func, p0, args=(wl.copy(), obs.copy()), full_output=1)
    return p


def get_fit(logwave_obs, flux_obs, andmask, logwave_model, flux_model):
    z, z_err, dists = move_rv(logwave_obs, flux_obs, andmask, logwave_model, flux_model)
    flux_obs_z = interpolate.UnivariateSpline((logwave_obs - z)[np.where(andmask == 0)], flux_obs[np.where(andmask == 0)], s=0)(logwave_model)

    wl = logwave_model
    kf = flux_obs_z
    of = flux_model
    flag, weights = log_flags(wl)
    p = fit_p3(wl[flag], kf[flag], of[flag])
    cont = polyval(p, wl) * kf
    # d = cross_correlation(cont[flag], of[flag])
    errs = cont - of

    return errs


def init_param(subclass):
    if subclass[0] == 'O':
        teff = 25000

    elif subclass[0] == 'B':
        teff = 17500
    elif subclass[0] == 'A':
        teff = 9900
    elif subclass[0] == 'F':
        teff = 6800
    elif subclass[0] == 'G':
        teff = 5500
    elif subclass[0] == 'K':
        teff = 4250
    elif subclass[0] == 'M':
        teff = 3000
    else:
        teff = 6200
    return teff, 4, -0.1


def run(n1, n2, path, param_path, n_snr):
    # path='/home/wangr/data/LAMOST/dr4_param/'
    dirlist = os.listdir(path)
    params = pd.read_csv(param_path, header=None)
    model = km.load_model('/home/wangr/code/python/cnn_logg/cal_err/gsn_phoenix_r1800_v2.model')
    logwave_model = np.arange(3.5798, 3.9445, 0.0001)
    output = []
    n = len(dirlist)
    for i in range(n1, n2):
        # print i
        fitsname = dirlist[i]
        specid = f_to_s.getSpecidFromName(fitsname)
        fitsfile = path + fitsname
        hdu = pf.open(fitsfile)
        z_1D = hdu[0].header['1D_Z']
        beginwave = hdu[0].header['CRVAL1']
        step = hdu[0].header['CD1_1']
        row = hdu[0].header['NAXIS1']
        logwave_obs = beginwave + step * np.arange(row)
        # wl = 10 ** logwave_obs
        flux = np.array(hdu[0].data[0])
        flux_obs = flux / np.mean(flux)
        andmask = np.array(hdu[0].data[3])
        # subclass = hdu[0].header['1D_SUBCL']
        specid_lasp, teff_lasp, logg_lasp, feh_lasp, rv_lasp = params.values[i:i + 1][0]
        # print specid
        # print "LASP parameter:", teff_lasp, logg_lasp, feh_lasp, rv_lasp
        # teff, logg, feh = init_param(subclass)
        teff_lasp = np.log10(teff_lasp)
        flux_model = model.predict(np.array([[teff_lasp, logg_lasp, feh_lasp, 0]])).reshape(3647, )
        errs = get_fit(logwave_obs, flux_obs, andmask, logwave_model, flux_model)
        # print outcome
        output.append(specid + ',' + ','.join([str(s) for s in errs]) + '\n')

    with open('/home/wangr/data/LAMOST/pixerrs/errs_snr%s_%s_phoenix_v2_%s_%s.csv' % (str(n_snr), str(n_snr + 10), str(n1), str(n2)), 'w') as save:
        save.writelines(output)


def data_clean(df):
    return df[(df > df.mean() - 3 * df.std()) & (df < df.mean() + 3 * df.std())]


def mustd(errsfile, n_snr):
    path = '/home/wangr/data/LAMOST/dr5/pix_errs/'
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
    print
    len(pix_std)
    print
    len(pix_errs)

    np.savetxt(path + 'errmean_snr%s_%s.csv' % (str(n_snr), str(n_snr + 10)), pix_errs, delimiter=',')
    np.savetxt(path + 'errstd_snr%s_%s.csv' % (str(n_snr), str(n_snr + 10)), pix_std, delimiter=',')


def multi(n_snr):
    path = '/home/wangr/data/LAMOST/dr4_param_snr%s_%s_fits/' % (str(n_snr), str(n_snr + 10))
    param_path = '/home/wangr/data/LAMOST/param_lasp/param_snr%s_%s.csv' % (str(n_snr), str(n_snr + 10))
    # n_snr = 10
    fitsfile = os.listdir(path)
    n = len(fitsfile) / 1000
    print
    n
    tasks = []
    if n > 0:
        ns1 = np.arange(n + 1) * 1000
        ns2 = np.append(ns1[1:], len(fitsfile))

        for k in range(n):
            n1 = ns1[k]
            n2 = ns2[k]
            print
            n1, n2
            tasks.append(Process(target=run, args=(n1, n2, path, param_path, n_snr)))
        for task in tasks:
            task.start()
        for task in tasks:
            task.join()
    if n == 0:
        tasks.append(Process(target=run, args=(0, len(fitsfile), path, param_path, n_snr)))
        for task in tasks:
            task.start()
        for task in tasks:
            task.join()


if __name__ == '__main__':
    # n_snr=100
    # multi(n_snr)
    # os.system('cat /home/wangr/data/LAMOST/pixerrs/errs_snr%s_%s_phoenix_v2_*.csv > /home/wangr/data/LAMOST/pixerrs/errs_snr%s_%s_phoenix_v2.csv'%(str(n_snr), str(n_snr + 10),str(n_snr), str(n_snr + 10)))
    # os.system('rm /home/wangr/data/LAMOST/pixerrs/errs_snr%s_%s_phoenix_v2_*.csv'%(str(n_snr), str(n_snr + 10)))
    for n_snr in np.arange(1, 11) * 10:
        mustd('phoenix_errs_snr%s_%s.csv' % (str(n_snr), str(n_snr + 10)), n_snr)
