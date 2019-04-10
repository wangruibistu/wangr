import keras.models as km
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def load_model(model_path):
    model = km.load_model(model_path)
    return model


def denormalize(result, if_cannon):
    # if if_cannon:
    #     path = '/home/wangr/data/LAMOST_cross/med_apogee_param_flux_final_cannon.csv'
    # else:
    #     path = '/home/wangr/data/LAMOST_cross/med_apogee_param_flux_final.csv'
    # data_all = pd.read_csv(path)
    # medid,teff,logg,M_H,alpha_m,rv,flux_0-1501
    path_info = '/home/wangr/data/LAMOST/rv_lamostmed_apogee.csv'
    path_flux = '/home/wangr/data/LAMOST/flux_medres_apogee_cannon.csv'
    data_flux = pd.read_csv(path_flux, header=None)
    data_flux.columns = ['fitsid'] + ["flux_" + str(i) for i in range(1602)]
    data_info = pd.read_csv(path_info)
    data_info = data_info[(data_info.TEFF >= 3500) &
                          (data_info.TEFF <= 5500) &
                          (data_info.LOGG > 0) &
                          (data_info.LOGG < 3.5) &
                          (data_info.M_H != -9999.) &
                          (data_info.ALPHA_M != -9999.)]
    data_param_flux = pd.merge(data_info, data_flux, on='fitsid')
    # df_flux = data_param_flux.iloc[:, -1602:]
    df_param = data_param_flux.loc[:, ['TEFF', 'LOGG', 'M_H', 'ALPHA_M']]
    mean = df_param.mean()
    std = df_param.std()
    print 'parameters mean: ', mean.values
    print mean.values.shape
    print 'parameters std: ', std.values
    print std.values.shape
    return result * std.values + mean.values


def analysis_results(y_test, y_pre):
    data = pd.DataFrame(
            columns = ['teff_test', 'logg_test', 'mh_test', 'am_test', 'teff_pre', 'logg_pre', 'mh_pre', 'am_pre'],
            data = np.hstack([y_test, y_pre]))
    print len(data)
    data = data[(data.teff_pre >= 3500) & (data.teff_pre <= 5500)&
                (data.logg_pre <= 3.5) & (data.logg_pre >= 0) &
                (data.mh_pre <= 0.5) & (data.mh_pre >= -3) &
                (data.am_pre <= 0.8) & (data.am_pre >= -0.4)]
    print len(data)

    plt.figure(figsize = (16, 7))

    plt.subplot(241)
    plt.plot(data.teff_test, data.teff_pre, '.')
    plt.plot([3500, 5600], [3500, 5600], 'r--')
    plt.xlabel('Teff_APOGEE')
    plt.ylabel('Teff_LAMOST')

    plt.subplot(242)
    plt.plot(data.logg_test, data.logg_pre, '.')
    plt.plot([0, 4], [0, 4], 'r--')
    plt.xlabel('logg_APOGEE')
    plt.ylabel('logg_LAMOST')

    plt.subplot(243)
    plt.plot(data.mh_test, data.mh_pre, '.')
    plt.plot([-3, 1], [-3, 1], 'r--')
    plt.xlabel('[M/H]_APOGEE')
    plt.ylabel('[M/H]_LAMOST')

    plt.subplot(244)
    plt.plot(data.am_test, data.am_pre, '.')
    plt.plot([-0.3, 0.5], [-0.3, 0.5], 'r--')
    plt.xlabel('[$\\alpha$/M]_APOGEE')
    plt.ylabel('[$\\alpha$/M]_LAMOST')

    ax5 = plt.subplot(245)
    plt.hist(data.teff_pre - data.teff_test, bins = 50)
    plt.text(0.1, 0.9, '$\mu$=%.2f\n$\sigma$=%.2f' % (
        np.mean(data.teff_pre - data.teff_test), np.std(data.teff_pre - data.teff_test)), fontsize = 14, ha = 'left',
            va = 'top', transform = ax5.transAxes)
    plt.xlim(-500, 500)
    plt.xlabel('$\Delta$Teff')

    ax6 = plt.subplot(246)
    plt.hist(data.logg_pre - data.logg_test, bins = 50)
    plt.text(0.1, 0.9, '$\mu$=%.2f\n$\sigma$=%.2f' % (
        np.mean(data.logg_pre - data.logg_test), np.std(data.logg_pre - data.logg_test)), fontsize = 14, ha = 'left',
            va = 'top', transform = ax6.transAxes)
    plt.xlim(-1, 1)
    plt.xlabel('$\Delta$logg')

    ax7 = plt.subplot(247)
    plt.hist(data.mh_pre - data.mh_test, bins = 50)
    plt.text(0.1, 0.9,
            '$\mu$=%.2f\n$\sigma$=%.2f' % (np.mean(data.mh_pre - data.mh_test), np.std(data.mh_pre - data.mh_test)),
            fontsize = 14, ha = 'left', va = 'top', transform = ax7.transAxes)
    plt.xlim(-0.5, 0.5)
    plt.xlabel('$\Delta$[M/H]')

    ax8 = plt.subplot(248)
    plt.hist(data.am_pre - data.am_test, bins = 50)
    plt.text(0.1, 0.9,
            '$\mu$=%.2f\n$\sigma$=%.2f' % (np.mean(data.am_pre - data.am_test), np.std(data.am_pre - data.am_test)),
            fontsize = 14, ha = 'left', va = 'top', transform = ax8.transAxes)
    plt.xlim(-0.2, 0.2)
    plt.xlabel('$\Delta$[$\\alpha$/M]')
    plt.subplots_adjust(hspace = 0.3, wspace = 0.3)
    plt.show()


def get_model_pre(model_path, df_flux):
    model = load_model(model_path)
    print model.summary()
    y_pre = model.predict(df_flux.reshape(len(df_flux), 1602, 1))
    return y_pre


def comp_apo_fparam_results():
    data=pd.read_csv('/home/wangr/data/LAMOST/medres_spec/medres_APOGEE_allcols.csv')
    data=data[(data.TEFF!=-9999.)&(data.LOGG!=-9999.)&(data.M_H!=-9999.)&(data.ALPHA_M!=-9999.)]
    data_fparam=np.array([s.replace(")","").replace("(","").replace("'",'').split(',') for s in data.FPARAM.values])
    cols=['Teff_f', 'logg_f', 'vmicro_f', 'M_H_f', 'C_M_f', 'N_M_f', 'a_M_f', 'vsini_f','O_M_f']
    for i in range(len(cols)):
        data.loc[:,cols[i]]=[float(s) for s in data_fparam[:,i]]
    return data

if __name__ == '__main__':
    model_structre = '22_cannon_t'
    if_cannon = True
    model_path = '/home/wangr/data/LAMOST/dnn_models/' + 'med_starnet_model_cnn_22_cannon_t.h5'
    # model=load_model(model_path)
    x_test = np.load('/home/wangr/data/LAMOST/dnn_test_sets/xtest_' + model_structre + '.npy')
    y_test = np.load('/home/wangr/data/LAMOST/dnn_test_sets/ytest_' + model_structre + '.npy')
    y_pre = get_model_pre(model_path, x_test)
    y_pre = denormalize(y_pre, if_cannon)
    y_test = denormalize(y_test, if_cannon)
    analysis_results(y_test, y_pre)
    data_f=comp_apo_fparam_results()
    data_f=data_f[['medres_specid','ASPCAP_ID','Teff_f', 'logg_f','M_H_f','a_M_f']]
    data=pd.read_csv('/home/wangr/data/LAMOST/medres_spec/ApoParam_medresFlux_blue_red_no9999.csv')
    data.columns=['medres_specid','teff','logg','feh','afe']+['flux_'+str(i) for i in range(8000)]
    data=pd.merge(data,data_f,on='medres_specid')