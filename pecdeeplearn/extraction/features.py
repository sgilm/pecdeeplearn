from __future__ import division

import numpy as np
import scipy.ndimage.interpolation


class FeatureError(Exception):
    """
    An error raised by feature functions when data cannot be extracted.

    Notes
        This error is used to communicate the idea that a particular feature
        could not be extracted at a particular point (and not necessarily that
        something is going wrong in the code).  It is useful for iterators
        that extract data using a try/except clause, to narrow down the scope
        of the except clause.

    """

    pass


def patch(volume, point, kernel_shape, prob_seg=False):
    """Fetch and process a patch of a specified size from given volume."""

    # Get the raw data as a 3D array.
    patch_data = raw_patch(volume, point, kernel_shape, prob_seg=prob_seg)

    return reshape_patch(patch_data, kernel_shape)


def flat_patch(volume, point, kernel_shape):
    """Fetch and return a flat patch of a specified size from given volume."""

    # Get the raw data as a 3D array.
    patch_data = raw_patch(volume, point, kernel_shape)

    return patch_data.reshape(np.prod(kernel_shape))


def scaled_patch(volume, point, source_kernel, target_kernel, prob_seg=False):
    """Get a patch of specified size and resample it to a different size."""

    # Get the scale factors and extract the patch.
    scale_factors = np.array(target_kernel) / np.array(source_kernel)
    patch_data = raw_patch(volume, point, source_kernel, prob_seg=prob_seg)
    scaled_data = scipy.ndimage.interpolation.zoom(patch_data, scale_factors)

    return reshape_patch(scaled_data, target_kernel)


def raw_patch(volume, point, kernel_shape, prob_seg=False):
    """Fetch raw data for a patch of specified size from a specified volume."""

    # Convert kernel shape to a mutable object for fixing up.
    kernel_shape = list(kernel_shape)

    # Ensure each kernel dimensions is odd, so it can be centred on a voxel.
    for i in range(len(kernel_shape)):
        if kernel_shape[i] % 2 == 0:
            kernel_shape[i] -= 1

    # The kernel must include 3 dimensions.
    if len(kernel_shape) != 3:
        raise FeatureError('The input kernel must include 3 dimensions.')

    # Create arrays to represent the kernel.
    kernel_starts = [-(size // 2) for size in kernel_shape]
    kernel_stops = [size // 2 + 1 for size in kernel_shape]
    kernel_ranges = [np.array([kernel_starts[i], kernel_stops[i]])
                     for i in range(len(kernel_shape))]

    # Create a list of patch frames as boundaries for extracting the patch.
    patch_frames = [list(kernel_ranges[i] + point[i])
                    for i in range(len(kernel_ranges))]

    # Check the validity of the patch frames.  If any indices are outside the
    # scope of the vols, raise an exception.
    for i in range(len(patch_frames)):

        # Check that each frame's starting element is >= 0.
        if patch_frames[i][0] < 0:
            raise FeatureError('Patch index too small.')

        # Check that each frame's ending element is <= max.
        elif patch_frames[i][1] > volume.shape[i]:
            raise FeatureError('Patch index too large.')

    # Create a tuple of the patch's indices.
    patch_indices = tuple([slice(*frame) for frame in patch_frames])

    if prob_seg:
        return volume.prob_seg_data[patch_indices]
    else:
        return volume.mri_data[patch_indices]


def reshape_patch(patch_data, kernel_shape):
    """Reshape raw patch data to be compatible with neural net input."""

    # Create the correct dimensions (patches are used in convolutions, so the
    # first dimension should always be 1 - for the number of colour channels).
    patch_shape = [1] + [size for size in kernel_shape if size != 1]

    return np.array(patch_data).reshape(patch_shape)


def intensity_mean(volume, point, kernel_shape):
    """Get the mean intensity of points within a patch of specified shape."""

    # Get the patch from the volume.
    voxel_patch = patch(volume, point, kernel_shape)

    return np.mean(voxel_patch)


def intensity_variance(volume, point, kernel_shape):
    """Get the variance of points within a patch of specified shape."""

    # Get the patch from the volume.
    voxel_patch = patch(volume, point, kernel_shape)

    return np.var(voxel_patch)


def landmark_displacement(volume, point, landmark_name):
    """Get the displacement of a point from a given landmark."""

    # Convert the coordinates of the point to an array for ease of use.
    coordinate_array = np.array(point)

    # Do the same for voxel spacings.
    spacing_array = np.array(volume.header.get_zooms()[:3])

    return coordinate_array * spacing_array - volume.landmarks[landmark_name]


def point_offset(volume, point, offset):
    """Get the intensity of a single voxel, offset from the specified point."""

    # Convert the quantities into arrays for ease of use.
    point_array = np.array(point)
    offset_array = np.array(offset)

    return volume.mri_data[tuple(point_array + offset_array)].reshape(1)
