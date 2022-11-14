import numpy as np

def load_volume(path, mode='optovue_foct_6m'):
    """loading OCT cube from raw data

    Parameters:
        path (str list) -- a single directory path for raw data
    """
    mode_info = {
        'optovue_foct_6m': {'suffix': '.foct', 'size': (400, 400, 640)},
        'optovue_ssada_6m': {'suffix': '.ssada', 'size': (400, 400, 160)},
        'optovue_foct_3m': {'suffix': '.foct', 'size': (304, 304, 640)},
        'optovue_ssada_3m': {'suffix': '.ssada', 'size': (304, 304, 160)},
        'cirrus_img_6m_128': {'suffix': '.img', 'size': (128, 1024, 512)},
        'cirrus_img_6m_200': {'suffix': '.img', 'size': (200, 1024, 200)}
    }
    info = mode_info[mode]

    assert path.endswith(info['suffix'])

    with open(path, 'rb') as load_f:
        if info['suffix'] == '.img':
            cube_data = np.fromfile(load_f, dtype='uint8').reshape(mode_info[mode]['size'])
            cube_data = cube_data[:, ::-1, ::-1]
        else:
            cube_data = np.fromfile(load_f, dtype='float32').reshape(mode_info[mode]['size']).transpose((0, 2, 1))
            cube_data = cube_data[:, ::-1, :]
    return normalize(cube_data)


def normalize(x, mode='max_min', s=0.1):
    """Normalize the image range for visualization"""

    if mode == 'max_min':
        out = (x - np.min(x)) / (np.max(x) - np.min(x))
    elif mode == 'mean_std':
        z = x / np.std(x)
        out = np.clip((z - z.mean()) / max(z.std, 1e-4) * s + 0.5, 0, 1)
    else:
        raise('mode error')
    return np.uint8(255 * out)