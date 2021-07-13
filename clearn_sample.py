import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('/home/wangr/data/isochrones/Gaia/Gaia_isochrones.csv')
param = pd.read_csv('/home/wangr/data/model_spectra_lib/phoenix/MedResFITS/param_r1800.csv')
out = []
for i in np.arange(0.5, 5.5, 0.5):
    if i == 0:
        min = 3.44369
        max = 3.64139
    elif i == 0.5:
        min = 3.49114
        max = 3.66709
    elif i == 1:
        min = 3.5287
        max = 3.69279
    elif i == 1.5:
        min = 3.55835
        max = 3.7264
    elif i == 2:
        min = 3.58999
        max = 3.85887
    elif i == 2.5:
        min = 3.61371
        max = 3.97156

    elif i == 3:
        min = 3.63151
        max = 4.08
    elif i == 3.5:
        min = 3.64535
        max = 4.12
    elif i == 4:
        min = 3.675
        max = 5
    elif i == 4.5:
        min = 3.58208
        max = 5
    elif i == 5:
        min = 3.44764
        max = 3.68489
    elif i == 5.5:
        min = 3.37449
        max = 3.56429
    elif i == 6:
        min = 0
        max = 3.5

    # min = data[data.logg == i]['logteff'].min()
    # max = data[data.logg == i]['logteff'].max()
    res = param[(param.logg == i) & (param.teff <= max) & (param.teff >= min) & (param.feh != -4)]
    for s in res.index:
        out.append(str(int(s)) + '\n')
with open('/home/wangr/data/index_range_new.csv', 'w') as save:
    save.writelines(out)

index = np.loadtxt('/home/wangr/data/index_range_new.csv')

param = param.iloc[index, :]

plt.figure()
plt.plot(data.logteff, data.logg, '.')
plt.plot(param.teff, param.logg, '.')
plt.gca().invert_xaxis()
plt.gca().invert_yaxis()
plt.show()
