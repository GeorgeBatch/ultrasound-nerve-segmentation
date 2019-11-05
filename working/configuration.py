# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

########################################################################################################################
# ======================================================================================================================
# configuration
# ======================================================================================================================
########################################################################################################################

# standard-module imports
from keras.optimizers import Adam

# separate-module imports
from check_pars import ALLOWED_PARS, check_dict_subset

# look up the format and the available parameters
print(ALLOWED_PARS)

# The result is very sensitive to the choice of the Learning Rate parameter  of the optimizer
# DO NOT CHANGE THE NAME, you can change the parameters
OPTIMIZER = Adam(lr=0.0045)

# DO NOT CHANGE THE NAME, you can change the parameters
PARS = {
    'outputs': 1,
    'activation': 'relu',
    'pooling_block': {'trainable': False},
    'information_block': {'convolution': {'simple': 'normalized'}},
    'connection_block': 'not_residual'
}

# DO NOT REMOVE THESE LINES, they checks if your parameter choice is valid
assert PARS.keys() == ALLOWED_PARS.keys()
print(check_dict_subset(PARS, ALLOWED_PARS))
