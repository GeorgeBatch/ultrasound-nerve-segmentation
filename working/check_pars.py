# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

########################################################################################################################
# ======================================================================================================================
# check_pars
# ======================================================================================================================
########################################################################################################################
# read-only file!!!

# standard-module imports
from keras.optimizers import Adam


def check_dict_subset(subset, superset):
    """Checks if one nested dictionary is a subset of another

    :param subset: subset dictionary
    :param superset: superset dictionary
    :return: if failed: gives helpful print statements and assertion error
             if successful, prints 'Your parameter choice is valid'
    """
    print("superset keys:", superset.keys())
    print("subset keys:", subset.keys())
    assert all(item in superset.keys() for item in subset.keys())
    print("Subset keys is a subset of superset keys", all(item in superset.keys() for item in subset.keys()))
    for key in subset.keys():
        print("superset key items:", superset[key])
        print("subset key items:", subset[key])
        if type(superset[key]) == dict:
            assert type(subset[key]) == type(superset[key])
            check_dict_subset(subset[key], superset[key])
        elif type(superset[key]) == list:
            assert subset[key] in superset[key]
            print("subset[key] item:", subset[key], " is in superset[key] items:", superset[key])
        else:
            print("Something went wrong. Uncomment the print statements in check_dict_subset() for easier debugging.")
            return type(superset[key]), superset[key]

    return 'Your parameter choice is valid'


# Only change ALLOWED_PARS if adding new functionality
ALLOWED_PARS = {
    'outputs': [1, 2],
    'activation': ['elu', 'relu'],
    'pooling_block': {
        'trainable': [True, False]},
    'information_block': {
        'inception': {
            'v1': ['a', 'b'],
            'v2': ['a', 'b', 'c'],
            'et': ['a', 'b']},
        'convolution': {
            'simple': ['not_normalized', 'normalized'],
            'dilated': ['not_normalized', 'normalized']}},
    'connection_block': ['not_residual', 'residual']
}

# for reference: in combination, these parameter choice showed the best performance
BEST_OPTIMIZER = Adam(lr=0.0045)
BEST_PARS = {
    'outputs': 2,
    'activation': 'elu',
    'pooling_block': {'trainable': True},
    'information_block': {'inception': {'v2': 'b'}},
    'connection_block': 'residual'
}

print(check_dict_subset(BEST_PARS, ALLOWED_PARS))
