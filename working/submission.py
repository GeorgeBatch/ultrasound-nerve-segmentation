# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

########################################################################################################################
# ======================================================================================================================
# Submission
# ======================================================================================================================
########################################################################################################################

# standard-module imports
import os
import numpy as np
from skimage.transform import resize
from itertools import chain

# separate-module imports
from configuration import PARS
from data import load_test_ids, image_rows, image_cols, _dir


def prep(img):
    """Prepare the image for to be used in a submission

    :param img: 2D image
    :return: resized version of an image
    """
    img = img.astype('float32')
    img = resize(img, (image_rows, image_cols), preserve_range=True)
    img = (img > 0.5).astype(np.uint8)  # threshold
    return img


def run_length_enc(label):
    """Create a run-length-encoding of an image

    :param label: image to be encoded
    :return: string with run-length-encoding of an image
    """
    x = label.transpose().flatten()
    y = np.where(x > 0)[0]

    # consider empty all masks with less than 10 pixels being greater than 0
    if len(y) < 10:
        return ''

    z = np.where(np.diff(y) > 1)[0]
    start = np.insert(y[z + 1], 0, y[0])
    end = np.append(y[z], y[-1])
    length = end - start
    res = [[s + 1, l + 1] for s, l in zip(list(start), list(length))]
    res = list(chain.from_iterable(res))
    return ' '.join([str(r) for r in res])


def submission():
    """Create a submission .csv file.

    The file will have 2 cols: img, pixels.
        The image column consists of the ids of test images.
        The pixels column consists of the run-length-encodings of the corresponding images.
    """
    imgs_id_test = load_test_ids()

    print('Loading imgs_test from imgs_mask_test.npy')
    imgs_test = np.load('imgs_mask_test.npy')
    if PARS['outputs'] == 2:
        print('Loading imgs_exist_test from imgs_mask_test_present.npy')
        imgs_exist_test = np.load('imgs_mask_test_present.npy')

    argsort = np.argsort(imgs_id_test)
    imgs_id_test = imgs_id_test[argsort]
    imgs_test = imgs_test[argsort]
    if PARS['outputs'] == 2:
        imgs_exist_test = imgs_exist_test[argsort]

    total = imgs_test.shape[0]
    ids = []
    rles = []  # run-length-encodings
    for i in range(total):
        img = imgs_test[i, :, :, 0]
        if PARS['outputs'] == 2:
            img_exist = imgs_exist_test[i]
        img = prep(img)

        # only for version with 2 outputs
        if PARS['outputs'] == 2:
            # new probability of nerve presence
            new_prob = (img_exist + min(1, np.sum(img) / 10000.0) * 5 / 3) / 2
            # setting mask to array of zeros if new probability of nerve presence < 0.5
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
