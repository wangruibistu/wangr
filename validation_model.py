import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

path = '/home/wangr/data/LAMOST/medres_spec/'
data = pd.read_csv(path + 'Apo_starNet_22_astroNN_blue_red_testResults.csv')
data.columns = ['teff_test', 'logg_test', 'mh_test', 'am_test', 'teff_pre', 'logg_pre', 'mh_pre', 'am_pre']

print len(data)
data = data[(data.teff_pre >= 3500) & (data.teff_pre <= 5500)&
            (data.logg_pre <= 3.5) & (data.logg_pre >= 0) &
            (data.mh_pre <= 0.5) & (data.mh_pre >= -3) &
            (data.am_pre <= 0.8) & (data.am_pre >= -0.1) &
            (data.am_test > -0.1)]
print len(data)
n = len(data)

#plt.figure(1, figsize = (12, 10))
#s1=plt.subplot(121)
#plt.scatter(data.teff_pre, data.logg_pre, facecolor = data.mh_pre.values, marker = '.', edgecolor = '')
#plt.text(0.1, 0.8, 'LAMOST-II\nN=%.d' % n, ha = 'left', va = 'top', fontsize = 14, transform = s1.transAxes)
#plt.xlim(3500, 5600)
#plt.ylim(0, 4)
#plt.gca().invert_xaxis()
#plt.gca().invert_yaxis()
#plt.xticks(fontsize = 14)
#plt.yticks(fontsize = 14)
#plt.xlabel('Teff(K)_LAMOST', fontsize = 14)
#plt.ylabel('log g_LAMOST', fontsize = 14)
#plt.colorbar()
#s2=plt.subplot(122)
#plt.scatter(data.teff_test, data.logg_test, facecolor = data.mh_test.values, marker = '.', edgecolor = '')
#plt.text(0.1, 0.8, 'APOGEE\nN=%.d' % n, ha = 'left', va = 'top', fontsize = 14, transform = s2.transAxes)
#plt.xlim(3500, 5600)
#plt.ylim(0, 4)
#plt.gca().invert_xaxis()
#plt.gca().invert_yaxis()
#plt.xticks(fontsize = 14)
#plt.yticks(fontsize = 14)
#plt.xlabel('Teff(K)_APOGEE', fontsize = 14)
#plt.ylabel('log g_APOGEE', fontsize = 14)
#plt.colorbar()

plt.figure(2,figsize=(16, 7))
plt.subplot(241)
plt.plot(data.teff_test, data.teff_pre, '.',color='k',alpha=0.5)
plt.plot([3500, 5600], [3500, 5600], 'r--')
plt.xlim(3500,5600)
plt.ylim(3500,5600)
plt.xlabel('Teff_APOGEE')
plt.ylabel('Teff_LAMOST')

plt.subplot(242)
plt.plot(data.logg_test, data.logg_pre, '.',color='k',alpha=0.5)
plt.plot([0, 4], [0, 4], 'r--')
plt.xlim(0,4)
plt.ylim(0,4)
plt.xlabel('logg_APOGEE')
plt.ylabel('logg_LAMOST')

plt.subplot(243)
plt.plot(data.mh_test, data.mh_pre, '.',color='k',alpha=0.5)
plt.plot([-3, 1], [-3, 1], 'r--')
plt.xlim(-3,1)
plt.ylim(-3,1)
plt.xlabel('[M/H]_APOGEE')
plt.ylabel('[M/H]_LAMOST')

plt.subplot(244)
plt.plot(data.am_test, data.am_pre, '.',color='k',alpha=0.5)
plt.plot([-0.1, 0.5], [-0.1, 0.5], 'r--')
plt.xlim(-0.1,0.42)
plt.ylim(-0.1,0.42)
plt.xlabel('[$\\alpha$/M]_APOGEE')
plt.ylabel('[$\\alpha$/M]_LAMOST')

ax5 = plt.subplot(245)
plt.hist(data.teff_pre - data.teff_test,  bins=100, facecolor='lightblue',edgecolor='white')
plt.text(0.1, 0.9,
    '$\mu$=%.2f\n$\sigma$=%.2f' % (np.mean(data.teff_pre - data.teff_test), np.std(data.teff_pre - data.teff_test)),
    fontsize=14, ha='left', va='top', transform=ax5.transAxes)
plt.xlim(-500, 500)
plt.xlabel('$\Delta$Teff')

ax6 = plt.subplot(246)
plt.hist(data.logg_pre - data.logg_test, bins=100, facecolor='lightblue',edgecolor='white')
plt.text(0.1, 0.9,
    '$\mu$=%.2f\n$\sigma$=%.2f' % (np.mean(data.logg_pre - data.logg_test), np.std(data.logg_pre - data.logg_test)),
    fontsize=14, ha='left', va='top', transform=ax6.transAxes)
plt.xlim(-1, 1)
plt.xlabel('$\Delta$logg')

ax7 = plt.subplot(247)
plt.hist(data.mh_pre - data.mh_test, bins=100, facecolor='lightblue',edgecolor='white')
plt.text(0.1, 0.9,
    '$\mu$=%.2f\n$\sigma$=%.2f' % (np.mean(data.mh_pre - data.mh_test), np.std(data.mh_pre - data.mh_test)),
    fontsize=14, ha='left', va='top', transform=ax7.transAxes)
plt.xlim(-0.5, 0.5)
plt.xlabel('$\Delta$[M/H]')

ax8 = plt.subplot(248)
plt.hist(data.am_pre - data.am_test, bins=100, facecolor='lightblue',edgecolor='white')
plt.text(0.1, 0.9,
    '$\mu$=%.2f\n$\sigma$=%.2f' % (np.mean(data.am_pre - data.am_test), np.std(data.am_pre - data.am_test)),
    fontsize=14, ha='left', va='top', transform=ax8.transAxes)
plt.xlim(-0.2, 0.2)
plt.xlabel('$\Delta$[$\\alpha$/M]')
plt.subplots_adjust(hspace=0.3, wspace=0.3)

print len(data.teff_pre)

print len(data.teff_test)
plt.show()


