import torch.nn as nn

def get_activations(active_type='relu'):

    # initialize activation
    if active_type == 'relu':
        activation = nn.ReLU(inplace=True)
    elif active_type == 'lrelu':
        activation = nn.LeakyReLU(0.2, inplace=True)
    elif active_type == 'prelu':
        activation = nn.PReLU()
    elif active_type == 'selu':
        activation = nn.SELU(inplace=True)
    elif active_type == 'tanh':
        activation = nn.Tanh()
    elif active_type == 'none':
        activation = None
    else:
        assert 0, "Unsupported activation: {}".format(active_type)

    return activation