from keras.models import Sequential
from keras.layers import Dense, InputLayer, Flatten, Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D, AveragePooling1D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np
import pandas as pd
import os
from keras.regularizers import l2
from sklearn.model_selection import train_test_split
import keras.losses

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
wave_new_b = np.arange(4950, 5350., 0.1)  #4000
wave_new_r = np.arange(6350, 6750., 0.1) #4000
wave = np.hstack([wave_new_b, wave_new_r])
num_train = 30000  # 6793
num_cv = 7997  # 1295
num_test = 30000  # 6793
num_flux = 8000
num_labels = 20
activation = 'relu'
initializer = 'he_normal'
input_shape = (None, num_flux, 1)
# num_filters = [4, 8, 16, 32, 64]
filter_length = 4
pool_length = 4
# num_hidden = [1024, 512, 256, 128, 64]

batch_size = 64
max_epochs = 100
lr = 0.01
# lr=0.0005
beta_1 = 0.9
beta_2 = 0.999
optimizer_epsilon = 1e-08
early_stopping_min_delta = 0.001
early_stopping_patience = 5
reduce_lr_factor = 0.5
# reuce_lr_epsilon = 0.0001
reuce_lr_epsilon = 0.00005
reduce_lr_patience = 2
# reduce_lr_min = 0.00008#
reduce_lr_min = 1e-8
loss_function = 'mse'
metrics = ['mse']


def drop_null(df):
    return df.replace([np.inf, -np.inf], np.nan).dropna(subset=[df.columns.values], how='any')

def normalize(res,df_param):
    mean = df_param.mean()
    std = df_param.std()
    # print 'parameters mean: ', mean.values
    # print mean.values.shape
    # print 'parameters std: ', std.values
    # print std.values.shape
    return res * std.values + mean.values

def load_sample(df_flux, df_param):
    mean_labels = df_param.mean()
    std_labels = df_param.std()
    normed_param = (df_param - mean_labels) / std_labels
    # print mean_labels
    # print std_labels
    X = df_flux.values
    y = normed_param.values
    # print X.shape,y.shape
    X_train_1, X_test, y_train_1, y_test = train_test_split(X, y,
        test_size=num_test)
    X_train, X_cv, y_train, y_cv = train_test_split(X_train_1, y_train_1,
        test_size=num_cv)
    n_train = len(X_train)
    n_test = len(X_test)
    n_cv = len(X_cv)
    X_train = X_train.reshape(n_train, num_flux, 1)
    X_test = X_test.reshape(n_test, num_flux, 1)
    X_cv = X_cv.reshape(n_cv, num_flux, 1)

    return X_train, y_train, X_test, y_test, X_cv, y_cv



def train_model(x_train, y_train, x_val, y_val):
    optimizer = Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, epsilon=optimizer_epsilon, decay=0.0)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=early_stopping_min_delta,
        patience=early_stopping_patience, verbose=2, mode='min')
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=reduce_lr_factor, epsilon=reuce_lr_epsilon,
        patience=reduce_lr_patience, min_lr=reduce_lr_min, mode='min', verbose=2)
    model = Sequential([
        InputLayer(batch_input_shape=input_shape),
        Conv1D(kernel_initializer=initializer, activation=activation, padding="same", filters=4,
            kernel_size=filter_length),
        Conv1D(kernel_initializer=initializer, activation=activation, padding="same", filters=2,
            kernel_size=filter_length),
        # Conv1D(kernel_initializer=initializer, activation=activation, padding="same",
        # filters=num_filters[2],kernel_size=filter_length),
        # Conv1D(kernel_initializer=initializer, activation=activation, padding="same",
        # filters=num_filters[3],kernel_size=filter_length),
        MaxPooling1D(pool_size=pool_length),
        # AveragePooling1D(pool_size=pool_length),
        Flatten(),
        # Dense(units=num_hidden[1], kernel_initializer=initializer, activation=activation),
        Dense(units=192, kernel_initializer=initializer, activation=activation),
        # Dropout(0.2),
        Dense(units=128, kernel_initializer=initializer, activation=activation),
        Dropout(0.2),
        # Dense(units=16, kernel_initializer=initializer, activation=activation),
        # Dropout(0.2),
        Dense(units=num_labels, activation="linear")])
    model.compile(optimizer=optimizer, loss=loss_function, metrics=metrics)
    print model.summary()
    model.fit(x_train, y_train,
            steps_per_epoch = num_train / batch_size,
            epochs = max_epochs,
            validation_data = (x_val, y_val),
            callbacks = [early_stopping, reduce_lr],
            validation_steps = num_cv / batch_size)
    return model


def save_model(model, model_name):
    # starnet_model = 'med_starnet_cnn_4_3.h5'
    model.save('/home/wangr/data/wr_data/LAMOST/dnn_models/' + model_name)
    print model_name + ' saved.'


def run_train(df_flux, df_param, net_structre):
    X_train, y_train, X_test, y_test, X_val, y_val = load_sample(df_flux, df_param)
    np.save('/home/wangr/data/wr_data/LAMOST/dnn_test_sets/xtrain_' + net_structre, X_train)
    np.save('/home/wangr/data/wr_data/LAMOST/dnn_test_sets/ytrain_' + net_structre, y_train)
    np.save('/home/wangr/data/wr_data/LAMOST/dnn_test_sets/xval_' + net_structre, X_val)
    np.save('/home/wangr/data/wr_data/LAMOST/dnn_test_sets/yval_' + net_structre, y_val)
    np.save('/home/wangr/data/wr_data/LAMOST/dnn_test_sets/xtest_' + net_structre, X_test)
    np.save('/home/wangr/data/wr_data/LAMOST/dnn_test_sets/ytest_' + net_structre, y_test)
    model = train_model(X_train, y_train, X_val, y_val)
    save_model(model, net_structre + '.h5')


if __name__ == '__main__':
    # path_info = '/home/wangr/data/wr_data/LAMOST_cross/rv_lamostmed_apogee_param_clean.csv'
    path_flux = '/home/wangr/data/LAMOST/medres_spec/parameters/medspec_apogee_payne_0_100000.csv'
    net_structre = 'payne_astronn'
    data_flux = pd.read_csv(path_flux, header=None)
    data_flux.columns = ['medid','Teff', 'Logg','vmic','CH', 'NH', 'OH', 'MgH', 'AlH', 'SiH', 'SH','KH', 'CaH', 'TiH',
        'CrH', 'MnH','FeH', 'NiH','CuH','C12C13', 'vmac'] + ["flux_" + str(i) for i in range(num_flux)]
    df_flux = data_flux.iloc[:, -8000:]
    df_param = data_flux.loc[:, ['Teff','Logg','vmic','CH', 'NH', 'OH', 'MgH', 'AlH', 'SiH', 'SH','KH', 'CaH',
                                       'TiH','CrH', 'MnH','FeH', 'NiH','CuH','C12C13', 'vmac']]
    # df_param = normalize(df_param_raw)
    print  df_flux.shape
    run_train(df_flux, df_param, net_structre)
