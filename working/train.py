########################################################################################################################
# ======================================================================================================================
# train
# ======================================================================================================================
########################################################################################################################

# standard-module imports
import numpy as np
from skimage.transform import resize
from keras.callbacks import ModelCheckpoint, EarlyStopping

# separate-module imports
from u_model import get_unet, IMG_COLS as img_cols, IMG_ROWS as img_rows
from data import load_train_data, load_test_data, load_nerve_presence
from configuration import PARS, OPTIMIZER


def preprocess(imgs, to_rows=None, to_cols=None):
    """Resize all images in a 4D tensor of images of the shape (samples, rows, cols, channels).

    :param imgs: a 4D tensor of images of the shape (samples, rows, cols, channels)
    :param to_rows: new number of rows for images to be resized to
    :param to_cols: new number of rows for images to be resized to
    :return: a 4D tensor of images of the shape (samples, to_rows, to_cols, channels)
    """
    if to_rows is None or to_cols is None:
        to_rows = img_rows
        to_cols = img_cols

    print(imgs.shape)
    imgs_p = np.ndarray((imgs.shape[0], to_rows, to_cols, imgs.shape[3]), dtype=np.uint8)
    for i in range(imgs.shape[0]):
        imgs_p[i, :, :, 0] = resize(imgs[i, :, :, 0], (to_rows, to_cols), preserve_range=True)
    return imgs_p


def train_and_predict():
    print('-' * 30)
    print('Loading and preprocessing train data...')
    print('-' * 30)
    imgs_train, imgs_mask_train = load_train_data()
    imgs_present = load_nerve_presence()

    imgs_train = preprocess(imgs_train)
    imgs_mask_train = preprocess(imgs_mask_train)

    # centering and standardising the images
    imgs_train = imgs_train.astype('float32')
    mean = np.mean(imgs_train)
    std = np.std(imgs_train)
    imgs_train -= mean
    imgs_train /= std

    imgs_mask_train = imgs_mask_train.astype('float32')
    imgs_mask_train /= 255.  # scale masks to be in {0, 1} instead of {0, 255}

    print('-' * 30)
    print('Creating and compiling model...')
    print('-' * 30)

    # load model - the Learning rate scheduler choice is most important here
    model = get_unet(optimizer=OPTIMIZER, pars=PARS)

    model_checkpoint = ModelCheckpoint('weights.h5', monitor='val_loss', save_best_only=True)
    early_stopping = EarlyStopping(patience=5, verbose=1)

    print('-' * 30)
    print('Fitting model...')
    print('-' * 30)

    if PARS['outputs'] == 1:
        imgs_labels = imgs_mask_train
    else:
        imgs_labels = [imgs_mask_train, imgs_present]

    model.fit(imgs_train, imgs_labels,
              batch_size=128, epochs=50,
              verbose=1, shuffle=True,
              validation_split=0.2,
              callbacks=[model_checkpoint, early_stopping])

    print('-' * 30)
    print('Loading and preprocessing test data...')
    print('-' * 30)
    imgs_test = load_test_data()
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

    if PARS['outputs'] == 1:
        np.save('imgs_mask_test.npy', imgs_mask_test)
    else:
        np.save('imgs_mask_test.npy', imgs_mask_test[0])
        np.save('imgs_mask_test_present.npy', imgs_mask_test[1])


# --------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    train_and_predict()
