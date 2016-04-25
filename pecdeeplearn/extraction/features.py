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


def patch(volume, point, kernel_shape):
    """Fetch a patch of a specified size from a specified volumes."""

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
    # scope of the volumes, raise an exception.
    for i in range(len(patch_frames)):

        # Check that each frame's starting element is >= 0.
        if patch_frames[i][0] < 0:
            raise FeatureError('Patch index too small.')

        # Check that each frame's ending element is <= max.
        elif patch_frames[i][1] > volume.shape[i]:
            raise FeatureError('Patch index too large.')

    # Create a tuple of the patch's indices.
    patch_indices = tuple([slice(*frame) for frame in patch_frames])

    # Extract patch data from volumes.
    patch_data = volume.mri_data[patch_indices]

    # Create dimensions of the patch array to return (patches are used in
    # convolutions, so the first dimension should always be 1 - for the number
    # of colour channels.
    patch_shape = [1] + [size for size in kernel_shape if size != 1]

    return np.array(patch_data).reshape(patch_shape)


def scaled_patch(volume, point, source_kernel, target_kernel):
    """Get a patch of specified size and resample it to a different size."""

    # Get the scale factors and extract the patch.
    scale_factors = np.array(target_kernel) / np.array(source_kernel)
    voxel_patch = patch(volume, point, source_kernel)

    return scipy.ndimage.interpolation.zoom(voxel_patch, scale_factors)


def intensity_mean(volume, point, kernel_shape):
    """Get the mean intensity of points within a patch of specified shape."""

    # Get the patch from the volumes.
    voxel_patch = patch(volume, point, kernel_shape)

    return np.mean(voxel_patch)


def intensity_variance(volume, point, kernel_shape):
    """Get the variance of points within a patch of specified shape."""

    # Get the patch from the volumes.
    voxel_patch = patch(volume, point, kernel_shape)

    return np.var(voxel_patch)


def landmark_displacement(volume, point, landmark_name):
    """Get the displacement of a point from a given landmark."""

    # Convert the coordinates of the point to an array for ease of use.
    coordinate_array = np.array(point)

    return volume.landmarks[landmark_name] - coordinate_array
