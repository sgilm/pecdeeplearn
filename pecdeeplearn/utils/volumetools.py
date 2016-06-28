from __future__ import division

import numpy as np

def standardise_volumes(volumes):
    """Creates a standardised dataset (mean = 0, s.d. = 1)."""

    # Accumulate sums required for mean/s.d. calculations, as well as a point
    # count.
    intensity_sum = np.uint64(0)
    squared_intensity_sum = np.uint64(0)
    point_count = np.uint64(0)

    # Iterate through all volumes to obtain these sums, using the correct
    # uint32 type to avoid overflow.
    for volume in volumes:
        current_mri_data = volume.mri_data.astype('uint64')
        intensity_sum += np.sum(current_mri_data)
        squared_intensity_sum += np.sum(current_mri_data**2)
        point_count += current_mri_data.size

    # Calculate mean and variance from the relevant sums.
    overall_mean = intensity_sum / point_count
    overall_std = \
        np.sqrt(squared_intensity_sum / point_count - overall_mean**2)

    # Cast these values to float32 so that the transformed arrays fit into
    # memory.
    overall_mean = overall_mean.astype('float32')
    overall_std = overall_std.astype('float32')

    # Apply transformation to standardise data.
    for volume in volumes:
        volume.mri_data = volume.mri_data - overall_mean
        volume.mri_data = volume.mri_data / overall_std


def dice_coefficient(first_array, second_array, margins=(0, 0, 0)):

    # Check the array dimensions match.
    if first_array.shape != second_array.shape:
        raise Exception('Array dimensions do not match.')

    # Create slices to use for extracting the inner part of the arrays.
    margined_slices = [slice(margin, max_size - margin)
                       for margin, max_size in zip(margins, first_array.shape)]

    # Extract the inner part (within the margins).
    margined_first_array = first_array[margined_slices]
    margined_second_array = second_array[margined_slices]

    # Find the number of nonzero points the arrays have in common.
    intersection_count = np.count_nonzero(
        np.logical_and(margined_first_array, margined_second_array))

    # Find the number of elements in the union of the array's nonzero points.
    union_count = np.count_nonzero(margined_first_array) + \
                  np.count_nonzero(margined_second_array)

    return 2 * intersection_count / union_count