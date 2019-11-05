# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

########################################################################################################################
# ======================================================================================================================
# metric
# ======================================================================================================================
########################################################################################################################
# needed for u_model

# standard-module imports
import numpy as np
from keras import backend as K  # tensorflow backend


def dice_coef(mask_1, mask_2, smooth=1):
    """Compute the dice coefficient between two equal-sized masks.

    Dice Coefficient: https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
    We need to add smooth, because otherwise 2 empty (all zeros) masks will throw an error instead of
    giving 1 as an output.

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
    Need smooth, because otherwise 2 empty (all zeros) masks will throw an error instead of giving 1 as an output.

    :param mask_1: first mask
    :param mask_2: second mask
    :param smooth: Smoothing parameter for dice coefficient
    :return: Smoothened dice coefficient between two equal-sized masks
    """
    tr = mask_1.flatten()
    pr = mask_2.flatten()
    return (2. * np.sum(tr * pr) + smooth) / (np.sum(tr) + np.sum(pr) + smooth)


# ----------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    a = np.random.random((420, 100))
    b = np.random.random((420, 100))
    res = np_dice_coef(a, b)
    print(res)
