import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde


def analysis_plot():
    path = '/home/wangr/data/LAMOST_cross/lasp_gann_gcs_new.csv'
    data = pd.read_csv(path)
    data = data[data.logg_1 > 0]
    n = len(data)
    teff_gcs = data.Teff_1
    teff_lasp = data.teff_lasp
    teff_gann = data.teff_gann

    logg_gcs = data.logg_1
    logg_lasp = data.logg_lasp
    logg_gann = data.logg_gann

    feh_gcs = data.__Fe_H_
    feh_lasp = data.feh_lasp
    feh_gann = data.feh_gann

    xy_gcs = np.vstack([teff_gcs, logg_gcs])
    z_gcs = gaussian_kde(xy_gcs)(xy_gcs)

    xy_gann = np.vstack([teff_gann, logg_gann])
    z_gann = gaussian_kde(xy_gann)(xy_gann)

    xy_lasp = np.vstack([teff_lasp, logg_lasp])
    z_lasp = gaussian_kde(xy_lasp)(xy_lasp)

    plt.figure(figsize = (10, 8), dpi = 120)

    mpl.rcParams['xtick.direction'] = 'in'
    mpl.rcParams['ytick.direction'] = 'in'

    plt.subplot(331)
    plt.scatter(teff_gcs, logg_gcs, c = z_gcs, s = 10, edgecolor = '')
    plt.text(7000, 3.00, 'GCS', fontsize = 14)
    plt.text(7000, 3.3, 'N=%.d' % n, fontsize = 14)
    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)
    plt.xlabel('Teff(K)', fontsize = 14)
    plt.ylabel('log g', fontsize = 14)

    plt.subplot(332)
    plt.scatter(teff_lasp, logg_lasp, c = z_lasp, s = 10, edgecolor = '')
    plt.text(7000, 2.45, 'LASP', fontsize = 14)
    plt.text(7000, 2.75, 'N=%.d' % n, fontsize = 14)
    plt.ylim(2, 5)
    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)
    plt.xlabel('Teff(K)', fontsize = 14)
    plt.ylabel('log g', fontsize = 14)

    plt.subplot(333)
    plt.scatter(teff_gann, logg_gann, c = z_gann, s = 10, edgecolor = '')
    plt.text(7000, 2.45, 'GANN', fontsize = 14)
    plt.text(7000, 2.75, 'N=%.d' % n, fontsize = 14)
    plt.ylim(2, 5)
    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)
    plt.xlabel('Teff(K)', fontsize = 14)
    plt.ylabel('log g', fontsize = 14)

    plt.subplot(334)
    xy_teff = np.vstack([teff_gcs, teff_lasp])
    z_teff = gaussian_kde(xy_teff)(xy_teff)
    plt.scatter(teff_gcs, teff_lasp, c = z_teff, s = 5, edgecolor = '')
    plt.plot([4000, 7500], [4000, 7500], c = 'r', linestyle = ':')
    plt.text(4200, 6850, '$\mu=%.2f$' % np.mean(teff_gcs - teff_lasp), fontsize = 14)
    plt.text(4200, 6450, '$\sigma=%.2f$' % np.std(teff_gcs - teff_lasp), fontsize = 14)
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)
    plt.xlabel('Teff_GCS', fontsize = 14)
    plt.ylabel('Teff_LASP', fontsize = 14)

    plt.subplot(335)
    xy_logg = np.vstack([logg_gcs, logg_lasp])
    z_logg = gaussian_kde(xy_logg)(xy_logg)
    plt.scatter(logg_gcs, logg_lasp, c = z_logg, s = 5, edgecolor = '')
    plt.plot([2, 6], [2, 6], c = 'r', linestyle = ':')
    plt.text(2.25, 5.3, '$\mu=%.2f$' % np.mean(logg_gcs - logg_lasp), fontsize = 14)
    plt.text(2.25, 4.9, '$\sigma=%.2f$' % np.std(logg_gcs - logg_lasp), fontsize = 14)
    plt.xlim(2, 6)
    plt.ylim(2, 6)
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)
    plt.xlabel('logg_GCS', fontsize = 14)
    plt.ylabel('logg_LASP', fontsize = 14)

    plt.subplot(336)
    xy_feh = np.vstack([feh_gcs, feh_lasp])
    z_feh = gaussian_kde(xy_feh)(xy_feh)
    plt.scatter(feh_gcs, feh_lasp, c = z_feh, s = 5, edgecolor = '')
    plt.plot([-1.1, 1.1], [-1.1, 1.1], c = 'r', linestyle = ':')
    plt.text(-1, 0.75, '$\mu=%.2f$' % np.mean(feh_gcs - feh_lasp), fontsize = 14)
    plt.text(-1, 0.55, '$\sigma=%.2f$' % np.std(feh_gcs - feh_lasp), fontsize = 14)
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)
    plt.xlabel('[Fe/H]_GCS', fontsize = 14)
    plt.ylabel('[Fe/H]_LASP', fontsize = 14)

    plt.subplot(337)
    xy_teff = np.vstack([teff_gcs, teff_gann])
    z_teff = gaussian_kde(xy_teff)(xy_teff)
    plt.scatter(teff_gcs, teff_gann, c = z_teff, s = 5, edgecolor = '')
    plt.plot([4000, 7500], [4000, 7500], c = 'r', linestyle = ':')
    plt.text(4200, 6850, '$\mu=%.2f$' % np.mean(teff_gann - teff_gcs), fontsize = 14)
    plt.text(4200, 6450, '$\sigma=%.2f$' % np.std(teff_gann - teff_gcs), fontsize = 14)
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)
    plt.xlabel('Teff_GCS', fontsize = 14)
    plt.ylabel('Teff_GANN', fontsize = 14)

    plt.subplot(338)
    xy_logg = np.vstack([logg_gcs, logg_gann])
    z_logg = gaussian_kde(xy_logg)(xy_logg)
    plt.scatter(logg_gcs, logg_gann, c = z_logg, s = 5, edgecolor = '')
    plt.plot([2, 6], [2, 6], c = 'r', linestyle = ':')
    plt.text(2.25, 5.3, '$\mu=%.2f$' % np.mean(logg_gann - logg_gcs), fontsize = 14)
    plt.text(2.25, 4.9, '$\sigma=%.2f$' % np.std(logg_gann - logg_gcs), fontsize = 14)
    plt.xlim(2, 6)
    plt.ylim(2, 6)
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)
    plt.xlabel('logg_GCS', fontsize = 14)
    plt.ylabel('logg_GANN', fontsize = 14)

    plt.subplot(339)
    xy_feh = np.vstack([feh_gcs, feh_gann])
    z_feh = gaussian_kde(xy_feh)(xy_feh)
    plt.scatter(feh_gcs, feh_gann, c = z_feh, s = 5, edgecolor = '')
    plt.plot([-1.1, 1.1], [-1.1, 1.1], c = 'r', linestyle = ':')
    plt.text(-1, 0.75, '$\mu=%.2f$' % np.mean(feh_gann - feh_gcs), fontsize = 14)
    plt.text(-1, 0.55, '$\sigma=%.2f$' % np.std(feh_gann - feh_gcs), fontsize = 14)
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)
    plt.xlabel('[Fe/H]_GCS', fontsize = 14)
    plt.ylabel('[Fe/H]_GANN', fontsize = 14)
    # plt.tight_layout(left=0.2, bottom=0.2, right=0.8, top=0.8,hspace=0.2,wspace=0.3)
    plt.subplots_adjust(hspace = 0.4, wspace = 0.4)

    # plt.savefig('/home/wangr/GCS_lasp.png', dpi=120)
    plt.show()


if __name__ == '__main__':
    # download_common()
    # get_lasp_param()
    # read_data()
    # data_merge()
    analysis_plot()
