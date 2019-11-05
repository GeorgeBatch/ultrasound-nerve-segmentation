# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

########################################################################################################################
# ======================================================================================================================
# u_model_blocks
# ======================================================================================================================
########################################################################################################################
# needed for u_model

# standard-module imports
from keras.layers import add, concatenate, Conv2D, MaxPooling2D
from keras.layers import BatchNormalization, Lambda
from keras.layers.advanced_activations import ELU, LeakyReLU


# ======================================================================================================================
# utility blocks needed for internal performance
# ======================================================================================================================

def NConv2D(filters, kernel_size, strides=(1, 1), padding='valid', dilation_rate=1,
            activation=None, kernel_initializer='glorot_uniform'):
    """Create a (Normalized Conv2D followed by a chosen activation) function
    Conv2D -> BatchNormalization -> activation()

    :param filters: Integer, the dimensionality of the output space (i.e. the number of output filters in the
    convolution)
    :param kernel_size: An integer or tuple/list of 2 integers, specifying the height and width of the 2D convolution
                        window. Can be a single integer to specify the same value for all spatial dimensions.
    :param strides: An integer or tuple/list of 2 integers, specifying the strides of the convolution along the height
                    and width. Can be a single integer to specify the same value for all spatial dimensions.
                    Specifying any stride value != 1 is incompatible with specifying any dilation_rate value != 1.
    :param padding: one of 'valid' or 'same' (case-insensitive), 'valid' by default to have the same as Conv2D
    :param dilation_rate: an integer or tuple/list of a single integer, specifying the dilation rate
                    to use for dilated convolution. Currently, specifying any dilation_rate value != 1
                    is incompatible with specifying any strides value != 1
    :param activation:  string, one of 'elu' or 'relu' or None (case-sensitive),
                        specifies activation function to be performed after BatchNormalization
    :param kernel_initializer: Initializer for the kernel weights matrix (see initializers in keras documentation)
    :return: a function, combined of 2D Convolution, followed by BatchNormalization across filters,
             and specified activation in that order
    """
    assert activation in ['relu', 'elu', None]
    # actv is a function, not a string, like activation
    actv = activation == 'relu' and (lambda: LeakyReLU(0.0)) or activation == 'elu' and (lambda: ELU(1.0)) or None

    def f(_input):
        conv = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                      dilation_rate=dilation_rate, kernel_initializer=kernel_initializer)(_input)
        norm = BatchNormalization(axis=3)(conv)
        return actv()(norm)

    return f


# needed for rblock (residual block)
def _shortcut(_input, residual):
    stride_width = _input._keras_shape[1] / residual._keras_shape[1]
    stride_height = _input._keras_shape[2] / residual._keras_shape[2]
    equal_channels = residual._keras_shape[3] == _input._keras_shape[3]

    shortcut = _input
    # 1 X 1 conv if shape is different. Else identity.
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        shortcut = Conv2D(filters=residual._keras_shape[3], kernel_size=(1, 1),
                          strides=(stride_width, stride_height),
                          kernel_initializer="he_normal", padding="valid")(_input)

    return add([shortcut, residual])


def rblock(inputs, filters, kernel_size, padding='valid', activation=None, scale=0.1):
    """Create a scaled Residual block connecting the down-path and the up-path of the u-net architecture

    Activations are scaled by a constant to prevent the network from dying. Usually is set between 0.1 and 0.3. See:
    https://towardsdatascience.com/a-simple-guide-to-the-versions-of-the-inception-network-7fc52b863202

    :param inputs: Input 4D tensor (samples, rows, cols, channels)
    :param filters: Integer, the dimensionality of the output space (i.e. the number of output convolution filters)
    :param kernel_size: An integer or tuple/list of 2 integers, specifying the height and width of the 2D convolution
                        window. Can be a single integer to specify the same value for all spatial dimensions.
    :param padding: one of 'valid' or 'same' (case-insensitive), 'valid' by default to have the same as Conv2D
    :param activation:  string, one of 'elu' or 'relu' or None (case-sensitive),
                        specifies activation function to use everywhere in the block
    :param scale: scaling factor preventing the network from dying out
    :return: 4D tensor (samples, rows, cols, channels) output of a residual block, given inputs
    """
    assert activation in ['relu', 'elu', None]
    # actv is a function, not a string, like activation
    actv = activation == 'relu' and (lambda: LeakyReLU(0.0)) or activation == 'elu' and (lambda: ELU(1.0)) or None

    residual = Conv2D(filters=filters, kernel_size=kernel_size, padding=padding)(inputs)
    residual = BatchNormalization(axis=3)(residual)
    residual = Lambda(lambda x: x * scale)(residual)
    res = _shortcut(inputs, residual)
    return actv()(res)


# ======================================================================================================================
# information blocks
# ======================================================================================================================

def convolution_block(inputs, filters, kernel_size=(3, 3), padding='valid', activation=None,
                      version='normalized', pars={}, allowed_pars={}):
    """Create a version of a convolution block.

    Versions: with and without batch-normalization after convolutions.

    :param inputs: Input 4D tensor (samples, rows, cols, channels)
    :param filters: Integer, the dimensionality of the output space (i.e. the number of output filters in the
                    convolution).
    :param kernel_size: An integer or tuple/list of 2 integers, specifying the height and width of the 2D convolution
                        window. Can be a single integer to specify the same value for all spatial dimensions.
    :param padding: one of 'valid' or 'same' (case-insensitive), 'valid' by default to have the same as Conv2D
    :param activation: string, specifies activation function to use everywhere in the block
    :param version: version of the convolution block, one of 'not_normalized', 'normalized' (case sensitive)
    :param pars: dictionary of parameters passed to u-net, determines the version, if this type of block is chosen
    :param allowed_pars: dictionary of all allowed to be passed to u-net parameters
    :return: 4D tensor (samples, rows, cols, channels) output of a convolution block, given inputs
    """
    assert activation in ['relu', 'elu', None]

    # checking that the allowed version names did not change in ALLOWED_PARS
    if allowed_pars != {}:
        assert allowed_pars.get('information_block').get('convolution').get('simple') == ['not_normalized',
                                                                                          'normalized']
    # keep version argument if need to use without PARS
    assert version in ['not_normalized', 'normalized']
    # setting the version from pars
    if pars.get('information_block').get('convolution').get('simple') is not None:
        version = pars.get('information_block').get('convolution').get('simple')

    if version == 'normalized':
        conv1 = NConv2D(filters=filters, kernel_size=kernel_size, activation=activation, padding=padding)(inputs)
        return NConv2D(filters=filters, kernel_size=kernel_size, activation=activation, padding=padding)(conv1)
    else:
        conv1 = Conv2D(filters=filters, kernel_size=kernel_size, activation=activation, padding=padding)(inputs)
        return Conv2D(filters=filters, kernel_size=kernel_size, activation=activation, padding=padding)(conv1)


def dilated_convolution_block(inputs, filters, kernel_size=(3, 3), padding='valid', activation=None,
                              version='normalized', pars={}, allowed_pars={}):
    """Create a version of a dilated-convolution block.

    Versions: with and without batch-normalization after dilated convolutions.

    See more about dilated convolutions:
    https://towardsdatascience.com/review-dilated-convolution-semantic-segmentation-9d5a5bd768f5

    :param inputs: Input 4D tensor (samples, rows, cols, channels)
    :param filters: Integer, the dimensionality of the output space (i.e. the number of output filters in the
                    convolution).
    :param kernel_size: An integer or tuple/list of 2 integers, specifying the height and width of the 2D convolution
                        window. Can be a single integer to specify the same value for all spatial dimensions.
    :param padding: one of 'valid' or 'same' (case-insensitive), 'valid' by default to have the same as Conv2D
    :param activation: string, specifies activation function to use everywhere in the block
    :param version: version of the dilated-convolution block, one of 'not_normalized', 'normalized' (case sensitive)
    :param pars: dictionary of parameters passed to u-net, determines the version, if this type of block is chosen
    :param allowed_pars: dictionary of all allowed to be passed to u-net parameters
    :return: 4D tensor (samples, rows, cols, channels) output of a dilated-convolution block, given inputs
    """
    assert activation in ['relu', 'elu', None]

    # checking that the allowed version names did not change in ALLOWED_PARS
    if allowed_pars != {}:
        assert allowed_pars.get('information_block').get('convolution').get('dilated') == ['not_normalized',
                                                                                           'normalized']
    # keep version argument if need to use without PARS
    assert version in ['not_normalized', 'normalized']
    # setting the version from pars
    if pars.get('information_block').get('convolution') is not None:
        version = pars.get('information_block').get('convolution')

    if version == 'normalized':
        conv1 = NConv2D(filters=filters, kernel_size=kernel_size, padding=padding,
                        dilation_rate=2, activation=activation)(inputs)
        return NConv2D(filters=filters, kernel_size=kernel_size, padding=padding,
                       dilation_rate=1, activation=activation)(conv1)
    else:
        conv1 = Conv2D(filters=filters, kernel_size=kernel_size, padding=padding,
                       dilation_rate=2, activation=activation)(inputs)
        return Conv2D(filters=filters, kernel_size=kernel_size, padding=padding,
                      dilation_rate=1, activation=activation)(conv1)


def inception_block_v1(inputs, filters, activation=None, version='b', pars={}, allowed_pars={}):
    """Create a version of v1 inception block described in:
    https://towardsdatascience.com/a-simple-guide-to-the-versions-of-the-inception-network-7fc52b863202

    Create an inception block described in v1, sections 'a' (for naive version), or 'b' (with dimension reduction)
    Each version has 4 verticals in their structure. See the link above.

    For all versions, verticals 1 and 2 of the block start with 2D convolution, which:
        reduces the number of input filters to next convolutions (to make computation cheaper)
        uses (1, 1) kernels, no Normalization
        is NOT normalized
        is followed by specified activation
    For all versions, verticals 1, 2, 3:
        the final convolution layer is not normalised and not activated since it will be dene after concatenation
    Vertical 4 is just a Conv2D. Its gets normalized and activated after being concatenated with
        outputs of other verticals.
    The concatenated output of the verticals is normalised and then activated with a given activation

    :param inputs: Input 4D tensor (samples, rows, cols, channels)
    :param filters: Integer, the dimensionality of the output space (i.e. the number of output filters in the
    convolution).
    :param activation: string, specifies activation function to use everywhere in the block
    :param version: version of inception block, one of 'a', 'b' (case sensitive)
    :param pars: dictionary of parameters passed to u-net, determines the version, if this type of block is chosen
    :param allowed_pars: dictionary of all allowed to be passed to u-net parameters
    :return: 4D tensor (samples, rows, cols, channels) output of an inception block, given inputs
    """

    assert filters % 16 == 0

    # checking that the allowed version names did not change in ALLOWED_PARS
    if allowed_pars != {}:
        assert allowed_pars.get('information_block').get('inception').get('v1') == ['a', 'b']
    # keep version argument if need to use without PARS
    assert version in ['a', 'b']
    # setting the version from pars
    if pars.get('information_block').get('inception').get('v1') is not None:
        version = pars.get('information_block').get('inception').get('v1')

    assert activation in ['relu', 'elu', None]
    # actv is a function, not a string, like activation
    actv = activation == 'relu' and (lambda: LeakyReLU(0.0)) or activation == 'elu' and (lambda: ELU(1.0)) or None

    # vertical 1
    if version == 'a':
        c1 = Conv2D(filters=filters // 8, kernel_size=(5, 5), padding='same', kernel_initializer='he_normal')(inputs)
    else:
        c1_1 = Conv2D(filters=filters // 16, kernel_size=(1, 1), padding='same',
                      activation=activation, kernel_initializer='he_normal')(inputs)
        c1 = Conv2D(filters=filters // 8, kernel_size=(5, 5), padding='same', kernel_initializer='he_normal')(c1_1)

    # vertical 2
    if version == 'a':
        c2 = Conv2D(filters=filters // 2, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal')(inputs)
    else:
        c2_1 = Conv2D(filters=filters // 8 * 3, kernel_size=(1, 1), padding='same',
                      activation=activation, kernel_initializer='he_normal')(inputs)
        c2 = Conv2D(filters=filters // 2, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal')(c2_1)

    # vertical 3
    p3_1 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(inputs)
    if version == 'b':
        c3 = Conv2D(filters=filters // 8, kernel_size=(1, 1), padding='same', kernel_initializer='he_normal')(p3_1)
    else:
        c3 = p3_1

    # vertical 4
    c4_1 = Conv2D(filters=filters // 4, kernel_size=(1, 1), padding='same', kernel_initializer='he_normal')(inputs)
    c4 = c4_1

    # concatenating verticals together, normalizing and applying activation
    result = concatenate([c1, c2, c3, c4], axis=3)
    result = BatchNormalization(axis=3)(result)
    result = actv()(result)
    return result


def inception_block_v2(inputs, filters, activation=None, version='b', pars={}, allowed_pars={}):
    """Create a version of v1 inception block described in:
    https://towardsdatascience.com/a-simple-guide-to-the-versions-of-the-inception-network-7fc52b863202

    Create an inception block described in v2, sections 'a', 'b', or 'c'
    Each version has 4 verticals in their structure. See the link above.

    For all versions, verticals 1 and 2 of the block start with 2D convolution, which:
        reduces the number of input filters to next convolutions (to make computation cheaper)
        uses (1, 1) kernels, no Normalization
        is NOT normalized
        is followed by specified activation
    For all versions, verticals 1, 2, 3:
        the middle convolutions use NConv2D with given activation, see its docstring
        the final convolution layer is not normalised and not activated since it will be dene after concatenation
    Vertical 4 is just a Conv2D. Its gets normalized and activated after being concatenated with
        outputs of other verticals.
    The concatenated output of the verticals is normalised and then activated with a given activation

    :param inputs: Input 4D tensor (samples, rows, cols, channels)
    :param filters: Integer, the dimensionality of the output space (i.e. the number of output filters in the
                    convolution).
    :param activation: string, specifies activation function to use everywhere in the block
    :param version: version of inception block, one of 'a', 'b', 'c' (case sensitive)
    :param pars: dictionary of parameters passed to u-net, determines the version, if this type of block is chosen
    :param allowed_pars: dictionary of all allowed to be passed to u-net parameters
    :return: 4D tensor (samples, rows, cols, channels) output of an inception block, given inputs
    """
    assert filters % 16 == 0

    # checking that the allowed version names did not change in ALLOWED_PARS
    if allowed_pars != {}:
        assert allowed_pars.get('information_block').get('inception').get('v2') == ['a', 'b', 'c']
    # keep version argument if need to use without PARS
    assert version in ['a', 'b', 'c']
    # setting the version from pars
    if pars.get('information_block').get('inception').get('v2') is not None:
        version = pars.get('information_block').get('inception').get('v2')

    assert activation in ['relu', 'elu', None]
    # actv is a function, not a string, like activation
    actv = activation == 'relu' and (lambda: LeakyReLU(0.0)) or activation == 'elu' and (lambda: ELU(1.0)) or None

    # vertical 1
    c1_1 = Conv2D(filters=filters // 16, kernel_size=(1, 1), padding='same',
                  activation=activation, kernel_initializer='he_normal')(inputs)
    if version == 'a':
        c1_2 = NConv2D(filters=filters // 8, kernel_size=3, padding='same',
                       activation=activation, kernel_initializer='he_normal')(c1_1)
        c1 = Conv2D(filters=filters // 8, kernel_size=3, padding='same', kernel_initializer='he_normal')(c1_2)
    elif version == 'b':
        c1_2 = NConv2D(filters=filters // 8, kernel_size=(1, 3), padding='same',
                       activation=activation, kernel_initializer='he_normal')(c1_1)
        c1_3 = NConv2D(filters=filters // 8, kernel_size=(3, 1), padding='same',
                       activation=activation, kernel_initializer='he_normal')(c1_2)
        c1_4 = NConv2D(filters=filters // 8, kernel_size=(1, 3), padding='same',
                       activation=activation, kernel_initializer='he_normal')(c1_3)
        c1 = Conv2D(filters=filters // 8, kernel_size=(3, 1), padding='same', kernel_initializer='he_normal')(c1_4)
    else:
        c1_2 = NConv2D(filters=filters // 8, kernel_size=(1, 3), padding='same',
                       activation=activation, kernel_initializer='he_normal')(c1_1)
        c1_3 = NConv2D(filters=filters // 8, kernel_size=3, padding='same',
                       activation=activation, kernel_initializer='he_normal')(c1_2)
        c1_41 = Conv2D(filters=filters // 8, kernel_size=(1, 3), padding='same', kernel_initializer='he_normal')(c1_3)
        c1_42 = Conv2D(filters=filters // 8, kernel_size=(3, 1), padding='same', kernel_initializer='he_normal')(c1_3)
        c1 = concatenate([c1_41, c1_42], axis=3)

    # vertical 2
    c2_1 = Conv2D(filters=filters // 8 * 3, kernel_size=(1, 1), padding='same',
                  activation=activation, kernel_initializer='he_normal')(inputs)
    if version == 'a':
        c2 = Conv2D(filters=filters // 2, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal')(c2_1)
    elif version == 'b':
        c2_2 = NConv2D(filters=filters // 2, kernel_size=(1, 3), padding='same',
                       activation=activation, kernel_initializer='he_normal')(c2_1)
        c2 = Conv2D(filters=filters // 2, kernel_size=(3, 1), padding='same', kernel_initializer='he_normal')(c2_2)
    else:
        c2_21 = Conv2D(filters=filters // 2, kernel_size=(1, 3), padding='same', kernel_initializer='he_normal')(c2_1)
        c2_22 = Conv2D(filters=filters // 2, kernel_size=(3, 1), padding='same', kernel_initializer='he_normal')(c2_1)
        c2 = concatenate([c2_21, c2_22], axis=3)

    # vertical 3
    p3_1 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(inputs)
    c3 = Conv2D(filters=filters // 8, kernel_size=(1, 1), padding='same', kernel_initializer='he_normal')(p3_1)

    # vertical 4
    c4 = Conv2D(filters=filters // 4, kernel_size=(1, 1), padding='same', kernel_initializer='he_normal')(inputs)

    # concatenating verticals together, normalizing and applying activation
    result = concatenate([c1, c2, c3, c4], axis=3)
    result = BatchNormalization(axis=3)(result)
    result = actv()(result)
    return result


def inception_block_et(inputs, filters, activation='relu', version='b', pars={}, allowed_pars={}):
    """Create an inception block with 2 options.
    For intuition read, parts v1 and v2:
    https://towardsdatascience.com/a-simple-guide-to-the-versions-of-the-inception-network-7fc52b863202

    Each version/option has 4 verticals in their structure. See the link above.
    Default option: version='b'
        Create an inception block close to one described in v2, but keeps 5 as a factor for some convolutions
    Alternative option: version='a'
        Create an inception block described in v1, section


    Function author Edward Tyantov. That's why the name: inception_block_et.
    My modifications

        use version='a' instead of split=False
        use version='b' instead of split=True

        change default to version='b', aka split=True

        swap: Conv2D -> BatchNormalization -> activation
        to:   NConv2D blocks. See NConv2D documentation for them.

        swap: Conv2D -> activation
        to:   Conv2D -> Conv2D(activation=activation)

        change the order of the verticals to coincide with v2_paper notation

        change names of the outputs of the block verticals to c1, c2, c3, c4

        use 'result' instead of 'res' to avoid confusion with residuals

    :param inputs: Input 4D tensor (samples, rows, cols, channels)
    :param filters: Integer, the dimensionality of the output space (i.e. the number of output filters in the
                    convolution).
    :param activation: activation function to use everywhere in the block
    :param version: version of inception block
    :param pars: dictionary of parameters passed to u-net, determines the version, if this type of block is chosen
    :param allowed_pars: dictionary of all allowed to be passed to u-net parameters
    :return: 4D tensor (samples, rows, cols, channels) output of an inception block, given inputs
    """
    assert filters % 16 == 0

    # checking that the allowed version names did not change in ALLOWED_PARS
    if allowed_pars != {}:
        assert allowed_pars.get('information_block').get('inception').get('et') == ['a', 'b']
    # keep version argument if need to use without PARS
    assert version in ['a', 'b']
    # setting the version from pars
    if pars.get('information_block').get('inception').get('et') is not None:
        version = pars.get('information_block').get('inception').get('et')

    assert activation in ['relu', 'elu', None]
    # actv is a function, not a string, like activation
    actv = activation == 'relu' and (lambda: LeakyReLU(0.0)) or activation == 'elu' and (lambda: ELU(1.0)) or None

    # vertical 1
    c1_1 = Conv2D(filters=filters // 16, kernel_size=(1, 1), padding='same',
                  activation=activation, kernel_initializer='he_normal')(inputs)
    if version == 'b':
        c1_2 = NConv2D(filters=filters // 8, kernel_size=(1, 5), padding='same',
                       activation=activation, kernel_initializer='he_normal')(c1_1)
        c1 = Conv2D(filters=filters // 8, kernel_size=(5, 1), kernel_initializer='he_normal', padding='same')(c1_2)
    else:
        c1 = Conv2D(filters=filters // 8, kernel_size=(5, 5), kernel_initializer='he_normal', padding='same')(c1_1)

    # vertical 2
    c2_1 = Conv2D(filters=filters // 8 * 3, kernel_size=(1, 1), padding='same',
                  activation=activation, kernel_initializer='he_normal')(inputs)
    if version == 'b':
        c2_2 = NConv2D(filters=filters // 2, kernel_size=(1, 3), padding='same',
                       activation=activation, kernel_initializer='he_normal')(c2_1)
        c2 = Conv2D(filters=filters // 2, kernel_size=(3, 1), kernel_initializer='he_normal', padding='same')(c2_2)
    else:
        c2 = Conv2D(filters=filters // 2, kernel_size=(3, 3), kernel_initializer='he_normal', padding='same')(c2_1)

    # vertical 3
    p3_1 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(inputs)
    c3 = Conv2D(filters=filters // 8, kernel_size=(1, 1), padding='same', kernel_initializer='he_normal')(p3_1)

    # vertical 4
    c4 = Conv2D(filters=filters // 4, kernel_size=(1, 1), padding='same', kernel_initializer='he_normal')(inputs)

    # concatenating verticals together, normalizing and applying activation
    result = concatenate([c1, c2, c3, c4], axis=3)
    result = BatchNormalization(axis=3)(result)
    result = actv()(result)
    return result


# ======================================================================================================================
# Combining blocks, allowing to use different blocks from before
# ======================================================================================================================

def pooling_block(inputs, filters, kernel_size=(3, 3), strides=(2, 2), padding='same', activation=None,
                  pool_size=(2, 2), trainable=True, pars={}, allowed_pars={}):
    """Function returning the output of one of the pooling blocks.

    Allows not to make different versions of the u-net in terms of how pooling operation is performed:
        1) trainable (default): through NConv2D custom function, see its documentation
        2) non-trainable (alternative): through MaxPooling operation

    To get the expected behaviour when changing 'trainable' assert strides == pool_size

    Parameters starting with p_ are only to be used for (trainable=False) MaxPooling2D
    Parameters starting with c_ are only to be used for (trainable=True) MaxPooling2D

    :param inputs: 4D tensor (samples, rows, cols, channels)
    :param filters:     NConv2D argument, filters
    :param kernel_size: NConv2D argument, kernel_size
    :param strides:     NConv2D argument, strides
    :param padding:     NConv2D/MaxPooling2D argument, padding
    :param activation:  NConv2D argument, activation
    :param pool_size:   MaxPooling2D argument, pool_size

    :param trainable: boolean specifying the version of a pooling block with default behaviour
        trainable=True: NConv2D(inputs._keras_shape[3], kernel_size=kernel_size, strides=strides, padding=padding)(
        inputs)
        trainable=False: MaxPooling2D(pool_size=pool_size)(inputs)
    :param pars: dictionary of parameters passed to u-net, determines the version of the block
    :param allowed_pars: dictionary of all allowed to be passed to u-net parameters

    :return: 4D tensor (samples, rows, cols, channels) output of a pooling block
    """
    # checking that the allowed trainable parameters did not change in ALLOWED_PARS
    if allowed_pars != {}:
        assert allowed_pars.get('pooling_block').get('trainable') == [True, False]
    # keep trainable argument if need to use without PARS
    assert trainable in [True, False]

    # setting the version from pars
    if pars.get('pooling_block').get('trainable') is not None:
        trainable = pars.get('pooling_block').get('trainable')

    # returning block's output
    if trainable:
        return NConv2D(filters=filters, kernel_size=kernel_size, strides=strides,
                       padding=padding, activation=activation)(inputs)
    else:
        return MaxPooling2D(pool_size=pool_size, padding=padding)(inputs)


def information_block(inputs, filters, kernel_size=(3, 3), padding='valid', activation=None,
                      block='inception', block_type='v2', version='b', pars={}, allowed_pars={}):
    """Function returning the output of one of the information blocks.

    :param inputs: 4D tensor (samples, rows, cols, channels)
    :param filters: Integer, the dimensionality of the output space (i.e. the number of output filters in the
                    convolution).
    :param kernel_size: An integer or tuple/list of 2 integers, specifying the height and width of the 2D convolution
                        window. Can be a single integer to specify the same value for all spatial dimensions.
    :param padding: one of 'valid' or 'same' (case-insensitive), 'valid' by default to have the same as Conv2D
    :param activation: string, specifies activation function to use everywhere in the block

    Next 3 parameters are there to be able to leave 'pars' and 'allowed_pars' empty
    :param block:       one of 'inception' or 'convolution' (case-sensitive)
    :param block_type:  if block == 'inception', one of 'v1', 'v2', 'et' (case-sensitive)
                        if block == 'convolution': one of 'simple', 'dilated' (case-sensitive)
    :param version:     version of a block to use

    :param pars: dictionary of parameters passed to u-net, determines the version of the block
    :param allowed_pars: dictionary of all allowed to be passed to u-net parameters

    :return: 4D tensor (samples, rows, cols, channels) output of a information block
    """
    # getting which block, block_type, version to use as the information block
    if pars.get('information_block') is not None:
        block = list(pars.get('information_block').keys())[0]
        block_type = list(pars.get('information_block').get(block).keys())[0]
        version = pars.get('information_block').get(block).get(block_type)

    # inception block
    if block == 'inception':
        if block_type == 'v1':
            return inception_block_v1(inputs=inputs, filters=filters, activation=activation,
                                      version=version, pars=pars, allowed_pars=allowed_pars)
        elif block_type == 'v2':
            return inception_block_v2(inputs=inputs, filters=filters, activation=activation,
                                      version=version, pars=pars, allowed_pars=allowed_pars)
        else:
            return inception_block_et(inputs=inputs, filters=filters, activation=activation,
                                      version=version, pars=pars, allowed_pars=allowed_pars)
    # convolution block
    else:
        if block_type == 'simple':
            return convolution_block(inputs=inputs, filters=filters, kernel_size=kernel_size,
                                     padding=padding, activation=activation,
                                     version=version, pars=pars, allowed_pars=allowed_pars)
        else:
            return dilated_convolution_block(inputs=inputs, filters=filters,
                                             kernel_size=kernel_size, padding=padding,
                                             activation=activation, version=version,
                                             pars=pars, allowed_pars=allowed_pars)


def connection_block(inputs, filters, padding='valid', activation=None,
                     version='residual', pars={}, allowed_pars={}):
    """Function returning the output of one of the connection block.

    :param inputs: 4D tensor (samples, rows, cols, channels)
    :param filters: Integer, the dimensionality of the output space (i.e. the number of output filters in the
                    convolution).
    :param padding: one of 'valid' or 'same' (case-insensitive), 'valid' by default to have the same as Conv2D
    :param activation:  string, one of 'elu' or 'relu' or None (case-sensitive),
                        specifies activation function to use everywhere in the block

    Version parameter is there to be able to leave 'pars' and 'allowed_pars' empty
    :param version: one of 'not_residual' or 'residual', version of a block to use

    :param pars: dictionary of parameters passed to u-net, determines the version of the block
    :param allowed_pars: dictionary of all allowed to be passed to u-net parameters

    :return: 4D tensor (samples, rows, cols, channels) output of a connection block
    """
    # checking that the allowed trainable parameters did not change in ALLOWED_PARS
    if allowed_pars != {}:
        assert allowed_pars.get('connection_block') == ['not_residual', 'residual']
    # keep trainable argument if need to use without PARS
    assert version in ['not_residual', 'residual']
    # setting the version from pars
    if pars.get('connection_block') is not None:
        version = pars.get('connection_block')

    if version == 'residual':
        return rblock(inputs=inputs, filters=32, kernel_size=(1, 1), padding='same', activation=activation)
    else:
        return Conv2D(filters=filters, kernel_size=(2, 2), padding=padding, kernel_initializer='he_normal')(inputs)
