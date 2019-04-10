from keras.models import Sequential
from keras.layers import Dense, InputLayer, Flatten, Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
wave_new_b = np.arange(4950, 5350, 0.1)  # 801
wave_new_r = np.arange(6350, 6700, 0.1)  # 701
# wave = np.hstack([wave_new_b, wave_new_r])
# num_flux = len(wave)  # n=3203
n_total = 40666
num_train = 15000
num_cv = 5666
num_test = 20000

num_flux = 4000


def drop_null(df):
    return df.replace([np.inf, -np.inf], np.nan).dropna(subset = [df.columns.values], how = 'any')


def load_sample(df_flux, df_param):
    mean_labels = df_param.mean()
    std_labels = df_param.std()
    normed_param = (df_param - mean_labels) / std_labels
    # print mean_labels
    # print std_labels
    X = df_flux.values
    y = normed_param.values
    print X.shape, y.shape
    X_train_1, X_test, y_train_1, y_test = train_test_split(X, y, test_size = num_test)
    X_train, X_cv, y_train, y_cv = train_test_split(X_train_1, y_train_1, test_size = num_cv)
    n_train = len(X_train)
    n_test = len(X_test)
    n_cv = len(X_cv)
    print n_train, n_test, n_cv
    X_train = X_train.reshape(n_train, num_flux, 1)
    X_test = X_test.reshape(n_test, num_flux, 1)
    X_cv = X_cv.reshape(n_cv, num_flux, 1)

    return X_train, y_train, X_test, y_test, X_cv, y_cv


def train_model(x_train, y_train, x_val, y_val):
    # num_flux = 1502
    num_labels = 4
    activation = 'relu'
    initializer = 'he_normal'
    input_shape = (None, num_flux, 1)
    num_filters = [4, 8, 16, 32]
    filter_length = 4
    pool_length = 4
    num_hidden = [1024, 512, 256, 128, 64]
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
    # reduce_lr_min = 0.00008  #
    reduce_lr_min = 1e-8
    loss_function = 'mse'
    metrics = ['mae']
    optimizer = Adam(lr = lr, beta_1 = beta_1, beta_2 = beta_2, epsilon = optimizer_epsilon, decay = 0.0)

    early_stopping = EarlyStopping(monitor = 'val_loss', min_delta = early_stopping_min_delta,
            patience = early_stopping_patience, verbose = 2, mode = 'min')

    reduce_lr = ReduceLROnPlateau(monitor = 'loss', factor = reduce_lr_factor, epsilon = reuce_lr_epsilon,
            patience = reduce_lr_patience, min_lr = reduce_lr_min, mode = 'min', verbose = 2)

    model = Sequential([InputLayer(batch_input_shape = input_shape),
        Conv1D(kernel_initializer = initializer, activation = activation, padding = "same", filters = num_filters[0],
                kernel_size = filter_length),
        Conv1D(kernel_initializer = initializer, activation = activation, padding = "same", filters = num_filters[1],
                kernel_size = filter_length),
        Conv1D(kernel_initializer = initializer, activation = activation, padding = "same", filters = num_filters[2],
                kernel_size = filter_length),
        Conv1D(kernel_initializer = initializer, activation = activation, padding = "same", filters = num_filters[3],
                kernel_size = filter_length), MaxPooling1D(pool_size = pool_length), Flatten(),
        Dense(units = num_hidden[0], kernel_initializer = initializer, activation = activation),
        Dense(units = num_hidden[1], kernel_initializer = initializer, activation = activation),
        Dense(units = num_hidden[2], kernel_initializer = initializer, activation = activation), # Dropout(0.2),
        Dense(units = num_hidden[3], kernel_initializer = initializer, activation = activation), Dropout(0.2),
        Dense(units = num_labels, activation = "linear", input_dim = num_hidden[3])])
    model.compile(optimizer = optimizer, loss = loss_function, metrics = metrics)
    model.fit(x_train, y_train, steps_per_epoch = num_train / batch_size, epochs = max_epochs,
            validation_data = (x_val, y_val), callbacks = [early_stopping, reduce_lr],
            validation_steps = num_cv / batch_size)  # ,max_q_size=10
    return model


def save_model(model, model_name):
    # starnet_model = 'med_starnet_cnn_4_3.h5'
    model.save('/home/wangr/data/LAMOST/dnn_models/' + model_name)
    print model_name + ' saved.'


if __name__ == '__main__':
    # path = '/home/wangr/data/LAMOST_cross/med_apogee_param_flux_final.csv'
    path = '/home/wangr/data/LAMOST/medres_spec/ApoParam_medresFlux_blue_red_no9999.csv'
    net_structre = '42_dropout'
    data_all = pd.read_csv(path,header=None)  # medid,teff,logg,M_H,alpha_m,rv,flux_0-1501  # 23565
    df_param = data_all.iloc[:, 1:5]  # param.columns=['teff','logg','M_H','A_M']
    df_flux = data_all.iloc[:, 5:4005]
    X_train, y_train, X_test, y_test, X_val, y_val = load_sample(df_flux, df_param)
    np.save('/home/wangr/data/LAMOST/dnn_test_sets/xtrain_' + net_structre, X_train)
    np.save('/home/wangr/data/LAMOST/dnn_test_sets/ytrain_' + net_structre, y_train)
    np.save('/home/wangr/data/LAMOST/dnn_test_sets/xval_' + net_structre, X_val)
    np.save('/home/wangr/data/LAMOST/dnn_test_sets/yval_' + net_structre, y_val)
    np.save('/home/wangr/data/LAMOST/dnn_test_sets/xtest_' + net_structre, X_test)
    np.save('/home/wangr/data/LAMOST/dnn_test_sets/ytest_' + net_structre, y_test)
    model = train_model(X_train, y_train, X_val, y_val)
    save_model(model, 'med_starnet_model_cnn_' + net_structre + '.h5')
