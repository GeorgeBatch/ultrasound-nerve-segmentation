# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

########################################################################################################################
# ======================================================================================================================
# data
# ======================================================================================================================
########################################################################################################################

# ======================================================================================================================
# Set-up
# ======================================================================================================================

# standard-module imports
import os
import numpy as np
from skimage.io import imread

# Input data files are available in the "../input/" directory.

_dir = os.path.abspath('')
os.chdir(_dir)
print(_dir)

print(os.listdir(_dir))
print(os.listdir("../input"))
print(os.listdir("../input/ultrasound-nerve-segmentation"))

# data
data_path = os.path.join('../input/ultrasound-nerve-segmentation', '')
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

# image dimensions
image_rows = 420
image_cols = 580


# ======================================================================================================================
# Functions for test and train data creation, storage and access
# ======================================================================================================================

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
        So for each train image saved, the patient number will be recorded exactly in the same order the
        images were saved.
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

        # With "continue" skip the mask image in the iteration because the mask will be saved together with
        # the image, when we get the image in one of the next iterations. This guarantees that the images,
        # masks and corresponding patient numbers are all saved in the correct order.
        if 'mask' in image_name:
            continue

        # we got to this point, meaning that image_name is a name of a training image and not a mask.

        # recreate the mask's name fot this image
        # noinspection PyTypeChecker
        image_mask_name = image_name.split('.')[0] + '_mask.tif'
        # get the patient number of the image
        patient_num = image_name.split('_')[0]
        # read the image itself to an np.array
        img = imread(os.path.join(train_data_path, image_name), as_gray=True)
        # read the corresponding mask to an np.array
        img_mask = imread(os.path.join(train_data_path, image_mask_name), as_gray=True)

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
    The np.array with test data ids will have shape (samples,). Each image id will be a number
    corresponding to the number in a test image name. For example image '8.tif' will have 8 as its image id.
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
        img = imread(os.path.join(test_data_path, image_name), as_gray=True)

        imgs[i, :, :, 0] = img
        imgs_id[i] = img_id

        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print('Loading done.')

    np.save(img_test_path, imgs)
    np.save(img_test_id_path, imgs_id)
    print('Saving to .npy files done.')


# ----------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    create_train_data()
    create_test_data()

# checking what is in the directory
print(os.listdir(preprocess_path))
