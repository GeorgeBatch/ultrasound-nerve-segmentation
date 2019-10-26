# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

# standard-module imports
import os

# Input data files are available in the "../input/" directory.

_dir = os.path.abspath('')
os.chdir(_dir)
print(_dir)

print(os.listdir(_dir))
print(os.listdir("../input"))
print(os.listdir("../input/ultrasound-nerve-segmentation"))

# data
data_path = os.path.join('/kaggle/input/ultrasound-nerve-segmentation', '')
preprocess_path = os.path.join(_dir, 'np_data')

if not os.path.exists(preprocess_path):
    os.mkdir(preprocess_path)
print(os.listdir(_dir))

# train data
img_train_path = os.path.join(preprocess_path, 'imgs_train.npy')
img_train_mask_path = os.path.join(preprocess_path, 'imgs_mask_train.npy')
img_train_patients = os.path.join(preprocess_path, 'imgs_patient.npy')
img_nerve_presence = os.path.join(preprocess_path, 'nerve_presence.npy')

# test data
img_test_path = os.path.join(preprocess_path, 'imgs_test.npy')
img_test_id_path = os.path.join(preprocess_path, 'imgs_id_test.npy')

print(os.listdir(preprocess_path))
# ====================================================================================================================
# Data
# ====================================================================================================================

# standard-module imports
import os
import numpy as np
import cv2

image_rows = 420
image_cols = 580


def load_test_data():
    """Load test data from a .npy file.

    :return: np.array with test data.
    """
    print('Loading test data from %s' % img_test_path)
    imgs_test = np.load(img_test_path)
    return imgs_test


def load_test_ids():
    """Load test ids from a .npy file.

    :return: np.array with test ids. Shape (samples, ).
    """
    print('Loading test ids from %s' % img_test_id_path)
    imgs_id = np.load(img_test_id_path)
    return imgs_id


def load_train_data():
    """Load train data from a .npy file.

    :return: np.array with train data.
    """
    print('Loading train data from %s and %s' % (img_train_path, img_train_mask_path))
    imgs_train = np.load(img_train_path)
    imgs_mask_train = np.load(img_train_mask_path)
    return imgs_train, imgs_mask_train


def load_patient_num():
    """Load the array with patient numbers from a .npy file

    :return: np.array with patient numbers
    """
    print('Loading patient numbers from %s' % img_train_patients)
    return np.load(img_train_patients)


def load_nerve_presence():
    """Load the array with binary nerve presence from a .npy file

    :return: np.array with patient numbers
    """
    print('Loading nerve presence array from %s' % img_nerve_presence)
    return np.load(img_nerve_presence)


def get_patient_nums(string):
    """Create a tuple (patient, photo) from image-file name patient_photo.tif

    :param string: image-file name in string format: patient_photo.tif
    :return: a tuple (patient, photo)

    >>> get_patient_nums('32_50.tif')
    (32, 50)
    """
    patient, photo = string.split('_')
    photo = photo.split('.')[0]
    return int(patient), int(photo)


def get_nerve_presence(mask_array):
    """Create an array specifying nerve presence on each of the masks in the mask_array

    :param mask_array: 4D tensor of a shape (samples, rows, cols, channels=1) with masks
    :return:
    """
    print("type(mask_array):", type(mask_array))
    print("mask_array.shape:", mask_array.shape)
    return np.array([int(np.sum(mask_array[i, :, :, 0]) > 0) for i in range(mask_array.shape[0])])


def create_train_data():
    """
    Create an np.array with patient numbers and save it into a .npy file.
    Create an np.array with train images and save it into a .npy file.
    Create an np.array with train masks and save it into a .npy file.

    The np.array with patient numbers will have shape (samples, ).
        So for each train image saved, the patient number will be recorded exactly in the same order the images were saved.
    The np.array with train images will have shape (samples, rows, cols, channels).
    The np.array with train masks will have shape (samples, rows, cols, channels).
        The masks are saved in the same order as the images.
    """
    train_data_path = os.path.join(data_path, 'train')
    images = os.listdir(train_data_path)
    total = len(images) // 2

    imgs = np.ndarray((total, image_rows, image_cols, 1), dtype=np.uint8)
    imgs_mask = np.ndarray((total, image_rows, image_cols, 1), dtype=np.uint8)
    i = 0
    print('Creating training images...')
    img_patients = np.ndarray((total,), dtype=np.uint8)
    for image_name in images:

        # With "continue" skip the mask image in the iteration because the mask will be saved together with the image,
        # when we get the image in one of the next iterations. This guarantees that the images, masks and corresponding
        # patient numbers are all saved in the correct order.
        if 'mask' in image_name:
            continue

        # we got to this point, meaning that image_name is a name of a training image and not a mask.

        # recreate the mask's name fot this image
        # noinspection PyTypeChecker
        image_mask_name = image_name.split('.')[0] + '_mask.tif'
        # get the patient number of the image
        patient_num = image_name.split('_')[0]
        # read the image itself to an np.array
        img = cv2.imread(os.path.join(train_data_path, image_name), cv2.IMREAD_GRAYSCALE)
        # read the corresponding mask to an np.array
        img_mask = cv2.imread(os.path.join(train_data_path, image_mask_name), cv2.IMREAD_GRAYSCALE)

        imgs[i, :, :, 0] = img
        imgs_mask[i, :, :, 0] = img_mask
        img_patients[i] = patient_num
        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print('Loading done.')

    # saving patient numbers, train images, train masks, nerve presence
    np.save(img_train_patients, img_patients)
    np.save(img_train_path, imgs)
    np.save(img_train_mask_path, imgs_mask)
    np.save(img_nerve_presence, get_nerve_presence(imgs_mask))

    print('Saving to .npy files done.')


def create_test_data():
    """
    Create an np.array with test data and save it into a .npy file.
    Create an np.array with ids for all images and save it into a .npy file.

    The np.array with test data will have shape (samples, rows, cols, channels).
    The np.array with test data ids will have shape (samples,). Each image id will be a number corresponding to the
    number in a test image name. For example image '8.tif' will have 8 as its image id.
    """
    test_data_path = os.path.join(data_path, 'test')
    images = os.listdir(test_data_path)
    total = len(images)

    imgs = np.ndarray((total, image_rows, image_cols, 1), dtype=np.uint8)
    imgs_id = np.ndarray((total,), dtype=np.int32)

    i = 0
    print('Creating test images...')
    for image_name in images:
        img_id = int(image_name.split('.')[0])
        img = cv2.imread(os.path.join(test_data_path, image_name), cv2.IMREAD_GRAYSCALE)

        imgs[i, :, :, 0] = img
        imgs_id[i] = img_id

        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print('Loading done.')

    np.save(img_test_path, imgs)
    np.save(img_test_id_path, imgs_id)
    print('Saving to .npy files done.')


# --------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    create_train_data()
    create_test_data()

# ====================================================================================================================
# metric - needed for u_model
# ====================================================================================================================

# standard-module imports
import numpy as np
from keras import backend as K  # tensorflow backend

smooth = 1


def dice_coef(mask_1, mask_2, smooth=1):
    """Compute the dice coefficient between two equal-sized masks.

    Dice Coefficient: https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
    We need to add smooth, because otherwise 2 empty (all zeros) masks will throw an error instead of giving 1 as an output.

    :param mask_1: first mask
    :param mask_2: second mask
    :param smooth: Smoothing parameter for dice coefficient
    :return: Smoothened dice coefficient between two equal-sized masks
    """
    mask_1_flat = K.flatten(mask_1)
    mask_2_flat = K.flatten(mask_2)

    # for pixel values in {0, 1} multiplication is the intersection of masks
    intersection = K.sum(mask_1_flat * mask_2_flat)
    return (2. * intersection + smooth) / (K.sum(mask_1_flat) + K.sum(mask_2_flat) + smooth)


def dice_coef_loss(mask_pred, mask_true):
    """Calculate dice coefficient loss, when comparing predicted mask for an image with the true mask

    :param mask_pred: predicted mask
    :param mask_true: true mask
    :return: dice coefficient loss
    """
    return -dice_coef(mask_pred, mask_true)


def np_dice_coef(mask_1, mask_2, smooth=1):
    """Compute the dice coefficient between two equal-sized masks.

    Used for testing on artificially generated np.arrays

    Dice Coefficient: https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
    We need to add smooth, because otherwise 2 empty (all zeros) masks will throw an error instead of giving 1 as an output.

    :param mask_1: first mask
    :param mask_2: second mask
    :param smooth: Smoothing parameter for dice coefficient
    :return: Smoothened dice coefficient between two equal-sized masks
    """
    tr = mask_1.flatten()
    pr = mask_2.flatten()
    return (2. * np.sum(tr * pr) + smooth) / (np.sum(tr) + np.sum(pr) + smooth)


# --------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    a = np.random.random((420, 100))
    b = np.random.random((420, 100))
    res = np_dice_coef(a, b)
    print(res)

# ====================================================================================================================
# u_model - needed for train
# ====================================================================================================================

# standard-module imports
import numpy as np
from keras.models import Model
from keras.layers import Input, add, concatenate, Conv2D, MaxPooling2D, UpSampling2D, Dense
from keras.layers import BatchNormalization, Dropout, Flatten, Lambda
from keras.layers.advanced_activations import ELU, LeakyReLU
from keras.optimizers import Adam

# # separate-module imports
# from metric import dice_coef, dice_coef_loss


IMG_ROWS, IMG_COLS = 80, 112
K.set_image_data_format('channels_last')  # (number of images, rows per image, cols per image, channels)


# --------------------------------------------------------------------------------------------------------------------
# Different blocks used for U-net
# --------------------------------------------------------------------------------------------------------------------

def inception_block(inputs, filters, split=False, activation='relu'):
    """Create an inception block with 2 options described in:
    https://towardsdatascience.com/a-simple-guide-to-the-versions-of-the-inception-network-7fc52b863202

    Default option: split=FALSE
        Create an inception block described in v1, section b

    Alternative option: split=TRUE
        Create an inception block described in v2

    :param inputs: Input 4D tensor (samples, rows, cols, channels)
    :param filters: Integer, the dimensionality of the output space (i.e. the number of output filters in the convolution).
    :param split: option of inception block
    :param activation: activation function to use everywhere in the block
    :return: output of the inception block, given inputs
    """
    assert filters % 16 == 0
    actv = activation == 'relu' and (lambda: LeakyReLU(0.0)) or activation == 'elu' and (lambda: ELU(1.0)) or None

    #
    # vertical 1
    #
    c1_1 = Conv2D(filters=filters // 4, kernel_size=(1, 1), kernel_initializer='he_normal', padding='same')(inputs)

    #
    # vertical 2
    #
    c2_1 = Conv2D(filters=filters // 8 * 3, kernel_size=(1, 1), kernel_initializer='he_normal', padding='same')(inputs)
    # no batch norm
    c2_1 = actv()(c2_1)
    if split:
        c2_2 = Conv2D(filters=filters // 2, kernel_size=(1, 3), kernel_initializer='he_normal', padding='same')(c2_1)
        c2_2 = BatchNormalization(axis=3)(c2_2)
        c2_2 = actv()(c2_2)
        c2_3 = Conv2D(filters=filters // 2, kernel_size=(3, 1), kernel_initializer='he_normal', padding='same')(c2_2)
    else:
        c2_3 = Conv2D(filters=filters // 2, kernel_size=(3, 3), kernel_initializer='he_normal', padding='same')(c2_1)

    #
    # vertical 3
    #
    c3_1 = Conv2D(filters=filters // 16, kernel_size=(1, 1), kernel_initializer='he_normal', padding='same')(inputs)
    # no batch norm
    c3_1 = actv()(c3_1)
    if split:
        c3_2 = Conv2D(filters=filters // 8, kernel_size=(1, 5), kernel_initializer='he_normal', padding='same')(c3_1)
        c3_2 = BatchNormalization(axis=3)(c3_2)  # mode=batch_mode # 0 in this case
        c3_2 = actv()(c3_2)
        c3_3 = Conv2D(filters=filters // 8, kernel_size=(5, 1), kernel_initializer='he_normal', padding='same')(c3_2)
    else:
        c3_3 = Conv2D(filters=filters // 8, kernel_size=(5, 5), kernel_initializer='he_normal', padding='same')(c3_1)

    #
    # vertical 4
    #
    p4_1 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(inputs)
    c4_2 = Conv2D(filters=filters // 8, kernel_size=(1, 1), kernel_initializer='he_normal', padding='same')(p4_1)

    #
    # concatenating verticals together
    #
    res = concatenate([c1_1, c2_3, c3_3, c4_2], axis=3)
    res = BatchNormalization(axis=3)(res)
    res = actv()(res)
    return res


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


def rblock(inputs, kernel_size, filters, scale=0.1):
    """Create a scaled Residual block connecting the down-path and the up-path of the u-net architecture

    Activations are scaled by a constant to prevent the network from dying. Usually is set between 0.1 and 0.3. See:
    https://towardsdatascience.com/a-simple-guide-to-the-versions-of-the-inception-network-7fc52b863202

    :param inputs: Input 4D tensor (samples, rows, cols, channels)
    :param filters: Integer, the dimensionality of the output space (i.e. the number of output filters in the convolution)
    :param kernel_size: An integer or tuple/list of 2 integers, specifying the height and width of the 2D convolution
                        window. Can be a single integer to specify the same value for all spatial dimensions.
    :param scale: scaling factor preventing the network from dying out
    :return: output of a residual block
    """
    residual = Conv2D(filters=filters, kernel_size=kernel_size, padding='same')(inputs)
    residual = BatchNormalization(axis=3)(residual)
    residual = Lambda(lambda x: x * scale)(residual)
    res = _shortcut(inputs, residual)
    return ELU()(res)


def NConv2D(filters, kernel_size, padding='same', strides=(1, 1)):
    """Create a (Normalized Conv2D followed by ELU activation) function
    Conv2D -> BatchNormalization -> ELU()

    :param filters: Integer, the dimensionality of the output space (i.e. the number of output filters in the
    convolution)
    :param kernel_size: An integer or tuple/list of 2 integers, specifying the height and width of the 2D convolution
                        window. Can be a single integer to specify the same value for all spatial dimensions.
    :param padding: one of "valid" or "same" (case-insensitive)
    :param strides: An integer or tuple/list of 2 integers, specifying the strides of the convolution along the height
                    and width. Can be a single integer to specify the same value for all spatial dimensions.
                    Specifying any stride value != 1 is incompatible with specifying any dilation_rate value != 1.
    :return: 2D Convolution function, followed by BatchNormalization across filters and ELU activation
    """

    def f(_input):
        conv = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides,
                      padding=padding)(_input)
        norm = BatchNormalization(axis=3)(conv)
        return ELU()(norm)

    return f


# --------------------------------------------------------------------------------------------------------------------
# Different U-net architectures
# --------------------------------------------------------------------------------------------------------------------


def get_unet_inception_2head(optimizer):
    """
    Creating and compiling the U-net
    :param optimizer: specifies the optimiser for u-net
    :return: compiled u-net model
    """

    split = True
    act = 'elu'

    #
    # down the U-net
    #

    inputs = Input((IMG_ROWS, IMG_COLS, 1), name='main_input')
    print("inputs:", inputs._keras_shape)
    conv1 = inception_block(inputs, 32, split=split, activation=act)
    print("conv1", conv1._keras_shape)
    # conv1 = inception_block(conv1, 32, split=split, activation=act)

    # pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    pool1 = NConv2D(32, kernel_size=(3, 3), padding='same', strides=(2, 2))(conv1)
    print("pool1", pool1._keras_shape)
    pool1 = Dropout(0.5)(pool1)
    print("pool1", pool1._keras_shape)

    conv2 = inception_block(pool1, 64, split=split, activation=act)
    print("conv2", conv2._keras_shape)
    # pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    pool2 = NConv2D(64, kernel_size=(3, 3), padding='same', strides=(2, 2))(conv2)
    print("pool2", pool2._keras_shape)
    pool2 = Dropout(0.5)(pool2)
    print("pool2", pool2._keras_shape)

    conv3 = inception_block(pool2, 128, split=split, activation=act)
    print("conv3", conv3._keras_shape)
    # pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    pool3 = NConv2D(128, kernel_size=(3, 3), padding='same', strides=(2, 2))(conv3)
    print("pool3", pool3._keras_shape)
    pool3 = Dropout(0.5)(pool3)
    print("pool3", pool3._keras_shape)

    conv4 = inception_block(pool3, 256, split=split, activation=act)
    print("conv4", conv4._keras_shape)
    # pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    pool4 = NConv2D(256, kernel_size=(3, 3), padding='same', strides=(2, 2))(conv4)
    print("pool4", pool4._keras_shape)
    pool4 = Dropout(0.5)(pool4)
    print("pool4", pool4._keras_shape)

    #
    # bottom level of the U-net
    #
    conv5 = inception_block(pool4, 512, split=split, activation=act)
    print("conv5", conv5._keras_shape)
    # conv5 = inception_block(conv5, 512, split=split, activation=act)
    conv5 = Dropout(0.5)(conv5)
    print("conv5", conv5._keras_shape)

    #
    # auxiliary head for predicting probability of nerve presence
    #
    pre = Conv2D(1, kernel_size=(1, 1), kernel_initializer='he_normal', activation='sigmoid')(conv5)
    pre = Flatten()(pre)
    aux_out = Dense(1, activation='sigmoid', name='aux_output')(pre)

    #
    # up the U-net
    #

    after_conv4 = rblock(conv4, 1, 256)
    print("after_conv4", after_conv4._keras_shape)
    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), after_conv4], axis=3)
    conv6 = inception_block(up6, 256, split=split, activation=act)
    print("conv6", conv6._keras_shape)
    conv6 = Dropout(0.5)(conv6)
    print("conv6", conv6._keras_shape)

    after_conv3 = rblock(conv3, 1, 128)
    print("after_conv3", after_conv3._keras_shape)
    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), after_conv3], axis=3)
    conv7 = inception_block(up7, 128, split=split, activation=act)
    print("conv7", conv7._keras_shape)
    conv7 = Dropout(0.5)(conv7)
    print("conv7", conv7._keras_shape)

    after_conv2 = rblock(conv2, 1, 64)
    print("after_conv2", after_conv2._keras_shape)
    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), after_conv2], axis=3)
    conv8 = inception_block(up8, 64, split=split, activation=act)  # batch_mode=2
    print("conv8", conv8._keras_shape)
    conv8 = Dropout(0.5)(conv8)
    print("conv8", conv8._keras_shape)

    after_conv1 = rblock(conv1, 1, 32)
    print("after_conv1", after_conv1._keras_shape)
    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), after_conv1], axis=3)
    conv9 = inception_block(up9, 32, split=split, activation=act)  # batch_mode=2
    print("conv9", conv9._keras_shape)
    # conv9 = inception_block(conv9, 32, split=split, activation=act) # batch_mode=2
    conv9 = Dropout(0.5)(conv9)
    print("conv9", conv9._keras_shape)

    # output
    conv10 = Conv2D(1, kernel_size=(1, 1), kernel_initializer='he_normal', activation='sigmoid', name='main_output')(
        conv9)
    print("conv10", conv10._keras_shape)

    # creating a model
    model = Model(inputs=inputs, outputs=[conv10, aux_out])

    # compiling the model
    model.compile(optimizer=optimizer,
                  loss={'main_output': dice_coef_loss, 'aux_output': 'binary_crossentropy'},
                  metrics={'main_output': dice_coef, 'aux_output': 'acc'},
                  loss_weights={'main_output': 1., 'aux_output': 0.5})

    return model


# --------------------------------------------------------------------------------------------------------------------
# get_unet() allows to try other versions of the u-net, if more are specified
get_unet = get_unet_inception_2head


# --------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    img_rows = IMG_ROWS
    img_cols = IMG_COLS

    # to check that model works without training, any kind of optimiser can be used
    model = get_unet(Adam(lr=1e-5))

    x = np.random.random((1, img_rows, img_cols, 1))
    res = model.predict(x, 1)
    print(res)
    print('params', model.count_params())
    print('layer num', len(model.layers))

# ====================================================================================================================
# Train
# ====================================================================================================================

from skimage.transform import resize
import numpy as np
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import backend as K

# Kaggle does not allow too many output files, so prediction masks are not saved
from skimage.io import imsave


# # TF dimension ordering in this code - use axis=3 in BatchNormalization()
K.set_image_data_format('channels_last')

img_rows = 80
img_cols = 112

smooth = 1.


def preprocess(imgs, to_rows=None, to_cols=None):
    if to_rows is None or to_cols is None:
        to_rows = img_rows
        to_cols = img_cols

    print(imgs.shape)
    imgs_p = np.ndarray((imgs.shape[0], to_rows, to_cols, imgs.shape[3]), dtype=np.uint8)
    for i in range(imgs.shape[0]):
        imgs_p[i, :, :, 0] = cv2.resize(imgs[i, :, :, 0], (to_cols, to_rows), interpolation=cv2.INTER_CUBIC)
    return imgs_p


def train_and_predict():
    print('-' * 30)
    print('Loading and preprocessing train data...')
    print('-' * 30)
    imgs_train, imgs_mask_train = load_train_data()
    imgs_present = load_nerve_presence()

    imgs_train = preprocess(imgs_train)
    imgs_mask_train = preprocess(imgs_mask_train)

    imgs_train = imgs_train.astype('float32')
    mean = np.mean(imgs_train)  # mean for data centering
    std = np.std(imgs_train)  # std for data normalization

    imgs_train -= mean
    imgs_train /= std

    imgs_mask_train = imgs_mask_train.astype('float32')
    imgs_mask_train /= 255.  # scale masks to be in {0, 1} instead of {0, 255}

    print('-' * 30)
    print('Creating and compiling model...')
    print('-' * 30)

    # load model - the Learning rate scheduler choice is most important here
    optimizer = Adam(lr=0.0045)
    model = get_unet(optimizer)

    model_checkpoint = ModelCheckpoint('weights.h5', monitor='val_loss', save_best_only=True)
    early_stopping = EarlyStopping(patience=5, verbose=1)

    print('-' * 30)
    print('Fitting model...')
    print('-' * 30)
    model.fit(imgs_train, [imgs_mask_train, imgs_present],
              batch_size=128, epochs=50,
              verbose=1, shuffle=True,
              validation_split=0.2,
              callbacks=[model_checkpoint, early_stopping])

    print('-' * 30)
    print('Loading and preprocessing test data...')
    print('-' * 30)
    imgs_test = load_test_data()
    imgs_id_test = load_test_ids()
    imgs_test = preprocess(imgs_test)

    imgs_test = imgs_test.astype('float32')
    imgs_test -= mean
    imgs_test /= std

    print('-' * 30)
    print('Loading saved weights...')
    print('-' * 30)
    model.load_weights('weights.h5')

    print('-' * 30)
    print('Predicting masks on test data...')
    print('-' * 30)

    imgs_mask_test = model.predict(imgs_test, verbose=1)

    np.save('imgs_mask_test.npy', imgs_mask_test[0])
    np.save('imgs_mask_test_present.npy', imgs_mask_test[1])


# --------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    train_and_predict()


# ====================================================================================================================
# Submission
# ====================================================================================================================

def prep(img):
    img = img.astype('float32')
    img = (img > 0.5).astype(np.uint8)  # threshold
    img = resize(img, (image_rows, image_cols), preserve_range=True)
    return img


def run_length_enc(label):
    from itertools import chain
    x = label.transpose().flatten()
    y = np.where(x > 0)[0]
    if len(y) < 10:  # consider as empty
        return ''
    z = np.where(np.diff(y) > 1)[0]
    start = np.insert(y[z + 1], 0, y[0])
    end = np.append(y[z], y[-1])
    length = end - start
    res = [[s + 1, l + 1] for s, l in zip(list(start), list(length))]
    res = list(chain.from_iterable(res))
    return ' '.join([str(r) for r in res])


def submission():
    # Uncomment the next line if the coede is separated in different files
    # from data import load_test_data

    imgs_id_test = load_test_ids()
    print('Loading imgs_test from imgs_mask_test.npy')
    imgs_test = np.load('imgs_mask_test.npy')
    print('Loading imgs_exist_test from imgs_mask_test_present.npy')
    imgs_exist_test = np.load('imgs_mask_test_present.npy')

    argsort = np.argsort(imgs_id_test)
    imgs_id_test = imgs_id_test[argsort]
    imgs_test = imgs_test[argsort]
    imgs_exist_test = imgs_exist_test[argsort]

    total = imgs_test.shape[0]
    ids = []
    rles = []
    for i in range(total):
        img = imgs_test[i, :, :, 0]
        img_exist = imgs_exist_test[i]
        img = prep(img)

        # new probability of nerve presence
        new_prob = (img_exist + min(1, np.sum(img) / 10000.0) * 5 / 3) / 2
        # setting mask to array of zeros if new probabiliry of nerve presence < 0.5
        if np.sum(img) > 0 and new_prob < 0.5:
            img = np.zeros((image_rows, image_cols))

        # producing run-length encoded version of the image
        rle = run_length_enc(img)

        rles.append(rle)
        ids.append(imgs_id_test[i])

        if i % 100 == 0:
            print('{}/{}'.format(i, total))

    # creating a submission file
    file_name = os.path.join(_dir, 'submission.csv')
    with open(file_name, 'w+') as f:
        f.write('img,pixels\n')
        for i in range(total):
            s = str(ids[i]) + ',' + rles[i]
            f.write(s + '\n')


# --------------------------------------------------------------------------------------------------------------------


if __name__ == '__main__':
    submission()

