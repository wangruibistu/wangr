import numpy as np
import pandas as pd
import six
from keras import backend as K
from keras.models import Model
from keras.layers.merge import add
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.layers import Dense, Flatten, Input, Activation
from keras.layers.convolutional import AveragePooling1D, Conv1D, MaxPooling1D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

num_train = 4500  # 6793
num_cv = 499  # 1295
num_test = 4000  # 6793
num_flux = 1602
num_labels = 4
activation = 'relu'
initializer = 'he_normal'
input_shape = (None, num_flux, 1)
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
    # print X.shape,y.shape
    X_train_1, X_test, y_train_1, y_test = train_test_split(X, y,
            test_size = float(num_test) / float(num_cv + num_train))
    X_train, X_cv, y_train, y_cv = train_test_split(X_train_1, y_train_1, test_size = float(num_cv) / float(num_train))
    n_train = len(X_train)
    n_test = len(X_test)
    n_cv = len(X_cv)
    X_train = X_train.reshape(n_train, num_flux, 1)
    X_test = X_test.reshape(n_test, num_flux, 1)
    X_cv = X_cv.reshape(n_cv, num_flux, 1)

    return X_train, y_train, X_test, y_test, X_cv, y_cv


def _bn_relu(input):
    """Helper to build a BN -> relu block
    """
    norm = BatchNormalization(axis = CHANNEL_AXIS)(input)
    return Activation("relu")(norm)


def _conv_bn_relu(**conv_params):
    """Helper to build a conv -> BN -> relu block
    """
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", 1)  # strides:conv step
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))

    def f(input):
        conv = Conv1D(filters = filters, kernel_size = kernel_size, strides = strides, padding = padding,
                kernel_initializer = kernel_initializer, kernel_regularizer = kernel_regularizer)(input)
        return _bn_relu(conv)

    return f


def _bn_relu_conv(**conv_params):
    """Helper to build a BN -> relu -> conv block.
    This is an improved scheme proposed in http://arxiv.org/pdf/1603.05027v2.pdf
    """
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", 1)
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))

    def f(input):
        print np.shape(input)
        activation = _bn_relu(input)
        return Conv1D(filters = filters, kernel_size = kernel_size, strides = strides, padding = padding,
                kernel_initializer = kernel_initializer, kernel_regularizer = kernel_regularizer)(activation)

    return f


def _shortcut(input, residual):
    """Adds a shortcut between input and residual block and merges them with "sum"
    """
    # Expand channels of shortcut to match residual.
    # Stride appropriately to match residual (width, height)
    # Should be int if network architecture is correctly configured.
    input_shape = K.int_shape(input)
    residual_shape = K.int_shape(residual)
    stride_width = int(round(input_shape[ROW_AXIS] / residual_shape[ROW_AXIS]))
    stride_height = int(round(input_shape[COL_AXIS] / residual_shape[COL_AXIS]))
    equal_channels = input_shape[CHANNEL_AXIS] == residual_shape[CHANNEL_AXIS]

    shortcut = input
    # 1 X 1 conv if shape is different. Else identity.
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        shortcut = Conv1D(filters = residual_shape[CHANNEL_AXIS], kernel_size = 1, strides = 1, padding = "valid",
                kernel_initializer = "he_normal", kernel_regularizer = l2(0.0001))(input)

    return add([shortcut, residual])


def _residual_block(block_function, filters, repetitions, is_first_layer = False):
    """Builds a residual block with repeating bottleneck blocks.
    """
    def f(input):
        for i in range(repetitions):
            init_strides = 1
            if i == 0 and not is_first_layer:
                init_strides = 2
            input = block_function(filters = filters, init_strides = init_strides,
                    is_first_block_of_first_layer = (is_first_layer and i == 0))(input)
        return input

    return f


def basic_block(filters, init_strides = 1, is_first_block_of_first_layer = False):
    """Basic 3 X 3 convolution blocks for use on resnets with layers <= 34.
    Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
    """
    def f(input):

        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            conv1 = Conv1D(filters = filters, kernel_size = 4, strides = init_strides, padding = "same",
                    kernel_initializer = "he_normal", kernel_regularizer = l2(1e-4))(input)
        else:
            conv1 = _bn_relu_conv(filters = filters, kernel_size = 4, strides = init_strides)(input)

        residual = _bn_relu_conv(filters = filters, kernel_size = 4)(conv1)
        return _shortcut(input, residual)

    return f


def bottleneck(filters, init_strides = 1, is_first_block_of_first_layer = False):
    """Bottleneck architecture for > 34 layer resnet.
    Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
    Returns:
        A final conv layer of filters * 4
    """

    def f(input):

        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            conv_1 = Conv1D(filters = filters, kernel_size = 1, strides = init_strides, padding = "same",
                    kernel_initializer = "he_normal", kernel_regularizer = l2(1e-4))(input)
        else:
            conv_1 = _bn_relu_conv(filters = filters, kernel_size = 1, strides = init_strides)(input)

        conv_4 = _bn_relu_conv(filters = filters, kernel_size = 4)(conv_1)
        residual = _bn_relu_conv(filters = filters * 4, kernel_size = 1)(conv_4)
        return _shortcut(input, residual)

    return f


def _handle_dim_ordering():
    global ROW_AXIS
    global COL_AXIS
    global CHANNEL_AXIS
    if K.image_dim_ordering() == 'tf':
        ROW_AXIS = 1
        COL_AXIS = 1
        CHANNEL_AXIS = 1
    else:
        CHANNEL_AXIS = 1
        ROW_AXIS = 1
        COL_AXIS = 1


def _get_block(identifier):
    if isinstance(identifier, six.string_types):
        res = globals().get(identifier)
        if not res:
            raise ValueError('Invalid {}'.format(identifier))
        return res
    return identifier


def build(input_shape, num_outputs, block_fn, repetitions):
    """Builds a custom ResNet like architecture.
    Args:
        input_shape: The input shape in the form (nb_channels, nb_rows, nb_cols)
        num_outputs: The number of outputs at final softmax layer
        block_fn: The block function to use. This is either `basic_block` or `bottleneck`.
            The original paper used basic_block for layers < 50
        repetitions: Number of repetitions of various block units.
            At each block unit, the number of filters are doubled and the input size is halved
    Returns:
        The keras `Model`.
    """
    # _handle_dim_ordering(self)
    # if len(input_shape) != 3:
    #     raise Exception("Input shape should be a tuple (nb_channels, nb_rows, nb_cols)")

    # Permute dimension order if necessary
    if K.image_dim_ordering() == 'tf':
        input_shape = (input_shape[0], input_shape[1], input_shape[2])

    # Load function from str if needed.
    block_fn = _get_block(block_fn)

    input = Input(shape = input_shape)
    conv1 = _conv_bn_relu(filters = 64, kernel_size = 4, strides = 1)(input)
    pool1 = MaxPooling1D(pool_size = 4, strides = 1, padding = "same")(conv1)

    block = pool1
    filters = 64
    for i, r in enumerate(repetitions):
        block = _residual_block(block_fn, filters = filters, repetitions = r, is_first_layer = (i == 0))(block)
        filters *= 2

    # Last activation
    block = _bn_relu(block)

    # Classifier block
    # block_shape = K.int_shape(block)
    pool2 = AveragePooling1D(pool_size = 4, strides = 1)(block)#(block_shape[ROW_AXIS], block_shape[COL_AXIS])
    flatten1 = Flatten()(pool2)
    dense = Dense(units = num_outputs, kernel_initializer = "he_normal", activation = "softmax")(flatten1)
    model = Model(inputs = input, outputs = dense)

    return model


class ResnetBuilder:
    def __init__(self, input_shape, num_outputs, block_fn):
        self.input_shape = input_shape
        self.num_outputs = num_outputs
        self.block_fn = block_fn  # self.repetitions = repetitions

    def build_resnet_18(self):
        self.model = build(self.input_shape, self.num_outputs, self.block_fn, [2, 2, 2])

    def build_resnet_34(self):
        self.model = build(self.input_shape, self.num_outputs, self.block_fn, [3, 4, 6])

    def build_resnet_50(self):
        self.model = build(self.input_shape, self.num_outputs, self.block_fn, [3, 4, 6])

    def build_resnet_101(self):
        self.model = build(self.input_shape, self.num_outputs, self.block_fn, [3, 4, 23])

    def build_resnet_152(self):
        self.model = build(self.input_shape, self.num_outputs, self.block_fn, [3, 8, 36])


if __name__ == '__main__':
    path_info = '/home/wangr/data/LAMOST_cross/rv_lamostmed_apogee.csv'
    path_flux = '/home/wangr/data/LAMOST/medres_spec/parameters/flux_medres_apogee_cannon.csv'
    net_structre = '22_cannon_64_4'
    data_flux = pd.read_csv(path_flux, header = None)
    df_fitsid = data_flux.iloc[:, 0:1]
    df_fitsid.columns = ['fitsid']
    df_flux = data_flux.iloc[:, 1:]
    data_info = pd.read_csv(path_info)
    data_param = pd.merge(df_fitsid, data_info, on = 'fitsid')
    print data_param.columns
    df_param = data_param.loc[:, ['TEFF', 'LOGG', 'M_H', 'ALPHA_M']]
    X_train, y_train, X_test, y_test, X_val, y_val = load_sample(df_flux, df_param)
    print np.shape(X_train)
    np.save('/home/wangr/data/LAMOST/dnn_test_sets/xtrain_' + net_structre, X_train)
    np.save('/home/wangr/data/LAMOST/dnn_test_sets/ytrain_' + net_structre, y_train)
    np.save('/home/wangr/data/LAMOST/dnn_test_sets/xval_' + net_structre, X_val)
    np.save('/home/wangr/data/LAMOST/dnn_test_sets/yval_' + net_structre, y_val)
    np.save('/home/wangr/data/LAMOST/dnn_test_sets/xtest_' + net_structre, X_test)
    np.save('/home/wangr/data/LAMOST/dnn_test_sets/ytest_' + net_structre, y_test)
    optimizer = Adam(lr = lr, beta_1 = beta_1, beta_2 = beta_2, epsilon = optimizer_epsilon, decay = 0.0)
    early_stopping = EarlyStopping(monitor = 'val_loss', min_delta = early_stopping_min_delta,
            patience = early_stopping_patience, verbose = 2, mode = 'min')
    reduce_lr = ReduceLROnPlateau(monitor = 'loss', factor = reduce_lr_factor, epsilon = reuce_lr_epsilon,
            patience = reduce_lr_patience, min_lr = reduce_lr_min, mode = 'min', verbose = 2)
    num_outputs = 4
    block_fn = basic_block(4, init_strides = 1, is_first_block_of_first_layer = False)
    print block_fn
    model_name = 'med_starnet_resnet_18.h5'
    resnet = ResnetBuilder(input_shape, num_outputs, block_fn)
    resnet.build_resnet_18()
    resnet.model.summary()
    resnet.model.compile(optimizer = optimizer, loss = loss_function, metrics = metrics)
    resnet.model.fit(X_train, y_train, steps_per_epoch = num_train / batch_size, epochs = max_epochs,
            validation_data = (X_val, y_val), callbacks = [early_stopping, reduce_lr],
            validation_steps = num_cv / batch_size)
    resnet.model.save('/home/wangr/data/LAMOST/medres_spec/parameters/dnn_model/' + model_name)
