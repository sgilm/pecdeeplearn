from __future__ import division

import numpy as np
from skimage import measure


def strip_connected_components(array, min_count=50000):

    label_array = measure.label(array)

    for label, count in zip(*np.unique(label_array, return_counts=True)):
        if count < min_count:
            indices = np.where(label_array == label)
            sample_point = tuple(indices[:, 0])
            array[indices] = 1 - array[sample_point]

    return array