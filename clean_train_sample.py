import pandas as pd
import keras.models as km
import numpy as np

# modelpath = '/home/wangr/data/LAMOST/dnn_models/gann_phoenix_n3461_noautoencoder_allsample.model'
modelpath = '/home/wangr/data/LAMOST/dnn_models/gsn_phoenix_n3461.model'
model = km.load_model(modelpath)
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


flag_cut = log_flags(logwave_model)
logwave_model = logwave_model[flag_cut]
wave_model = 10 ** logwave_model
blue_tmp = np.where(wave_model < 5800)
red_tmp = np.where(wave_model > 5800)


def pred_all_sample():
    path = '/home/wangr/data/model_spectra_lib/phoenix/MedResFITS/'
    flux = pd.read_csv(path + 'flux_r1800.csv', header=None)
    param = pd.read_csv(path + 'param_r1800.csv', header=None)
    param.columns = ['teff', 'logg', 'feh', 'alpha_fe']
    out = []

    for i in range(27704):
        # print '**************************************'
        teff, logg, feh, alpha_fe = param[i:i + 1].values[0]
        # print format(10 ** teff, '.2f'), logg, feh, alpha_fe
        flux_raw = flux[i:i + 1].values[0]
        flux_pre = model.predict(teff, logg, feh, alpha_fe, flux_raw)
        output=','.join(flux_pre)
        out.append(output)

    with open('/home/wangr/code/python/parameters/phoenix/selftest_phoenix_nopolyfit_r1800_v2.csv', 'w') as save:
        save.writelines(out)
