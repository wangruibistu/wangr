# coding=utf-8
import numpy as np
import pandas as pd
from keras.utils import plot_model
from keras.metrics import mae
from keras.models import Model
from keras.layers.merge import add
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.layers import Dense, Flatten, Input
from keras.layers.convolutional import AveragePooling1D, Conv1D, MaxPooling1D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from random import randint

num_total = 4502
num_train = 3000  # 6793
num_cv = 502  # 1295
num_test = 1000  # 6793
num_flux = 1602
num_labels = 4
activation = 'relu'
initializer = 'he_normal'
input_shape = (num_flux, 1)
num_filters = [4, 8, 16, 32, 64]
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
# reduce_lr_min = 0.00008#
reduce_lr_min = 1e-8
loss_function = 'mse'
metrics = ['mse']
num_outputs = 4


def generate_batch_data_random(x, y, batch_size):
    ylen = len(y)
    loopcount = ylen // batch_size
    while (True):
        i = randint(0, loopcount)
        yield x[i * batch_size:(i + 1) * batch_size], y[i * batch_size:(i + 1) * batch_size]


def drop_null(df):
    return df.replace([np.inf, -np.inf], np.nan).dropna(subset=[df.columns.values], how='any')


def load_sample(df_flux, df_param):
    mean_labels = df_param.mean()
    std_labels = df_param.std()
    normed_param = (df_param - mean_labels) / std_labels
    X = df_flux.values
    y = normed_param.values
    X_train_1, X_test, y_train_1, y_test = train_test_split(X, y,
                                                            test_size=float(num_test) / float(
                                                                num_cv + num_train + num_test))
    X_train, X_cv, y_train, y_cv = train_test_split(X_train_1, y_train_1,
                                                    test_size=float(num_cv) / float(num_train + num_cv))
    n_train = len(X_train)
    n_test = len(X_test)
    n_cv = len(X_cv)
    X_train = X_train.reshape(n_train, num_flux, 1)
    X_test = X_test.reshape(n_test, num_flux, 1)
    X_cv = X_cv.reshape(n_cv, num_flux, 1)

    return X_train, y_train, X_test, y_test, X_cv, y_cv


def Conv1d_BN(x, nb_filter, kernel_size, strides=1, padding='same', name=None):
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None

    x = Conv1D(nb_filter, kernel_size, padding=padding, strides=strides, activation='relu', name=conv_name)(x)

    x = BatchNormalization(axis=1, name=bn_name)(x)
    return x


def identity_Block(inpt, nb_filter, kernel_size, strides=1, with_conv_shortcut=False):
    x = Conv1d_BN(inpt, nb_filter=nb_filter, kernel_size=kernel_size, strides=strides, padding='same')
    x = Conv1d_BN(x, nb_filter=nb_filter, kernel_size=kernel_size, padding='same')
    if with_conv_shortcut:
        shortcut = Conv1d_BN(inpt, nb_filter=nb_filter, strides=strides, kernel_size=kernel_size)
        x = add([x, shortcut])
        return x
    else:
        x = add([x, inpt])
        return x


def bottleneck_Block(inpt, nb_filters, strides=1, with_conv_shortcut=False):
    k1, k2, k3 = nb_filters
    x = Conv1d_BN(inpt, nb_filter=k1, kernel_size=1, strides=strides, padding='same')
    x = Conv1d_BN(x, nb_filter=k2, kernel_size=4, padding='same')
    x = Conv1d_BN(x, nb_filter=k3, kernel_size=1, padding='same')
    if with_conv_shortcut:
        shortcut = Conv1d_BN(inpt, nb_filter=k3, strides=strides, kernel_size=1)
        x = add([x, shortcut])
        return x
    else:
        x = add([x, inpt])
        return x


def resnet_34():
    inpt = Input(shape=input_shape)
    # x = ZeroPadding1D(3))(inpt)
    x = inpt
    # conv1
    x = Conv1d_BN(x, nb_filter=64, kernel_size=8, strides=2, padding='valid')
    x = MaxPooling1D(pool_size=4, strides=2, padding='same')(x)

    # conv2_x
    x = identity_Block(x, nb_filter=64, kernel_size=4)
    x = identity_Block(x, nb_filter=64, kernel_size=4)
    x = identity_Block(x, nb_filter=64, kernel_size=4)

    # conv3_x
    x = identity_Block(x, nb_filter=128, kernel_size=4, strides=2, with_conv_shortcut=True)
    x = identity_Block(x, nb_filter=128, kernel_size=4)
    x = identity_Block(x, nb_filter=128, kernel_size=4)
    x = identity_Block(x, nb_filter=128, kernel_size=4)

    # conv4_x
    x = identity_Block(x, nb_filter=256, kernel_size=4, strides=2, with_conv_shortcut=True)
    x = identity_Block(x, nb_filter=256, kernel_size=4)
    x = identity_Block(x, nb_filter=256, kernel_size=4)
    x = identity_Block(x, nb_filter=256, kernel_size=4)
    x = identity_Block(x, nb_filter=256, kernel_size=4)
    x = identity_Block(x, nb_filter=256, kernel_size=4)

    # conv5_x
    x = identity_Block(x, nb_filter=512, kernel_size=4, strides=2, with_conv_shortcut=True)
    x = identity_Block(x, nb_filter=512, kernel_size=4)
    x = identity_Block(x, nb_filter=512, kernel_size=4)
    x = identity_Block(x, nb_filter=512, kernel_size=4)
    x = AveragePooling1D(pool_size=4)(x)

    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(num_outputs, activation='linear')(x)

    model = Model(inputs=inpt, outputs=x)
    return model


def resnet_50():
    inpt = Input(shape=input_shape)
    # x = ZeroPadding1D((3, 1))(inpt)
    x = inpt
    x = Conv1d_BN(x, nb_filter=64, kernel_size=8, strides=2, padding='valid')
    x = MaxPooling1D(pool_size=4, strides=2, padding='same')(x)
    # conv2_x
    x = bottleneck_Block(x, nb_filters=[64, 64, 256], strides=1, with_conv_shortcut=True)
    x = bottleneck_Block(x, nb_filters=[64, 64, 256])
    # x = bottleneck_Block(x, nb_filters=[64, 64, 256])

    # conv3_x
    x = bottleneck_Block(x, nb_filters=[128, 128, 512], strides=1, with_conv_shortcut=True)
    x = bottleneck_Block(x, nb_filters=[128, 128, 512])
    # x = bottleneck_Block(x, nb_filters=[128, 128, 512])
    # x = bottleneck_Block(x, nb_filters=[128, 128, 512])

    # conv4_x
    x = bottleneck_Block(x, nb_filters=[256, 256, 1024], strides=2, with_conv_shortcut=True)
    x = bottleneck_Block(x, nb_filters=[256, 256, 1024])
    # x = bottleneck_Block(x, nb_filters=[256, 256, 1024])
    # x = bottleneck_Block(x, nb_filters=[256, 256, 1024])
    # x = bottleneck_Block(x, nb_filters=[256, 256, 1024])
    # x = bottleneck_Block(x, nb_filters=[256, 256, 1024])

    # conv5_x
    x = bottleneck_Block(x, nb_filters=[512, 512, 2048], strides=2, with_conv_shortcut=True)
    x = bottleneck_Block(x, nb_filters=[512, 512, 2048])
    # x = bottleneck_Block(x, nb_filters=[512, 512, 2048])

    x = AveragePooling1D(pool_size=4)(x)
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(num_outputs, activation='linear')(x)

    model = Model(inputs=inpt, outputs=x)

    return model


def acc_top2(y_true, y_pred):
    return mae(y_true, y_pred)


if __name__ == '__main__':
    # if os.path.exists('resnet_50.h5'):
    #     model = load_model('resnet_50.h5')
    # else:
    #     model = check_print()
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
    df_flux = data_param_flux.iloc[:, -1602:]
    df_param = data_param_flux.loc[:, ['TEFF', 'LOGG', 'M_H', 'ALPHA_M']]
    print  df_flux.shape
    X_train, y_train, X_test, y_test, X_val, y_val = load_sample(df_flux, df_param)
    print np.shape(X_train)

    optimizer = Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, epsilon=optimizer_epsilon, decay=0.0)
    # early_stopping = EarlyStopping(monitor = 'val_loss', min_delta = early_stopping_min_delta,
    #         patience = early_stopping_patience, verbose = 2, mode = 'min')
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=reduce_lr_factor, epsilon=reuce_lr_epsilon,
                                  patience=reduce_lr_patience, min_lr=reduce_lr_min, mode='min', verbose=2)

    model = resnet_50()
    model.summary()
    # Save a PNG of the Model Build
    plot_model(model, to_file='resnet.png')
    model.compile(optimizer='adam', loss='mse', metrics=['mae', mae])
    print 'Model Compiled'
    # model.fit(X_train, y_train, steps_per_epoch = num_train / batch_size, epochs = max_epochs,
    #         validation_data = (X_val, y_val), callbacks = [early_stopping, reduce_lr],
    #         validation_steps = num_cv / batch_size)
    model.fit_generator(generate_batch_data_random(X_train, y_train, batch_size),
                        steps_per_epoch=len(y_train) // batch_size, nb_epoch=max_epochs,
                        validation_data=generate_batch_data_random(X_val, y_val, batch_size),
                        validation_steps=len(y_val) // batch_size)  # , callbacks = [early_stopping]

    net_structre = 'cannon_resnet_50_1d'
    np.save('/home/wangr/data/LAMOST/dnn_test_sets/xtrain_' + net_structre, X_train)
    np.save('/home/wangr/data/LAMOST/dnn_test_sets/ytrain_' + net_structre, y_train)
    np.save('/home/wangr/data/LAMOST/dnn_test_sets/xval_' + net_structre, X_val)
    np.save('/home/wangr/data/LAMOST/dnn_test_sets/yval_' + net_structre, y_val)
    np.save('/home/wangr/data/LAMOST/dnn_test_sets/xtest_' + net_structre, X_test)
    np.save('/home/wangr/data/LAMOST/dnn_test_sets/ytest_' + net_structre, y_test)
    model.save('/home/wangr/data/LAMOST/dnn_models/' + net_structre + '.h5')
