import numpy as np


def patch(volume, point, kernel_shape):
    # Ensure the kernel dimensions are odd, so it can be centred on a voxel.
    for i in range(len(kernel_shape)):
        if kernel_shape[i] % 2 == 0:
            kernel_shape[i] -= 1

    # Extend the kernel shape if fewer dimensions are given.
    if len(kernel_shape) < 3:
        kernel_shape += [1] * (3 - len(kernel_shape))
    num_dims = len(kernel_shape)

    kernel_starts = [-(size // 2) for size in kernel_shape]
    kernel_stops = [size // 2 + 1 for size in kernel_shape]
    kernel_ranges = [np.array([kernel_starts[i], kernel_stops[i]])
                     for i in range(num_dims)]

    patch_frames = [list(kernel_ranges[i] + point[i]) for i in range(num_dims)]

    for i in range(num_dims):
        if patch_frames[i][0] < 0:
            raise Exception
        elif patch_frames[i][1] > volume.shape[i]:
            raise Exception

    patch_indices = tuple([slice(*frame) for frame in patch_frames])
    return np.array(volume.mri_data[patch_indices])


def intensity_mean(volume, point, kernel_shape):
    patch = patch(volume, point, kernel_shape)
    mean_voxel_value = np.mean(patch)
    return mean_voxel_value


def intensity_variance(volume, point, kernel_shape):
    patch = patch(volume, point, kernel_shape)
    voxel_variance = np.var(patch)
    return voxel_variance