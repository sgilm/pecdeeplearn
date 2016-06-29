from __future__ import division

import numpy as np
import random


def probability_bins(volumes, num_bins=25, scale=None):
    """
    Create intensity bins and probabilities that a voxel in a bin is segmented.

    Args
        vols (list): a list of Volume objects that are to be used for
            creating the bins and probabilities.  All elements should be of the
            same size.
        num_bins (int): the number of bins to create.
        scale (float): the value to scale the probabilities to.  E.g. if
            scale = 1, then all probabilities in prob_bins are scaled by the
            same factor so that max(prob_bins) = 1.

    Returns
        bins (np.array): num_bins + 1 floats that define the boundaries of each
            bin
        prob_bins (np.array): num_bins floats that give the probability of a
            voxel that belongs to a particular bin being segmented

    """

    # Strip mri and seg data into separate arrays.
    all_mri_data = np.array([volume.mri_data for volume in volumes])
    all_seg_data = np.array([volume.seg_data for volume in volumes])

    # Get the total number of voxels in each bin, and the bin boundaries.
    voxel_counts, bins = np.histogram(all_mri_data, bins=num_bins)

    # Create a corresponding vols with each element being a bin label.
    voxel_bin_indices = np.digitize(all_mri_data, bins)

    # Strip away only those elements that are segmented.
    seg_bin_indices = voxel_bin_indices * all_seg_data

    # Count the number of segmented elements in each bin (discard first count,
    # as this corresponds to non-segmented elements).
    seg_counts = np.bincount(seg_bin_indices.flatten())[1:]

    # Lengthen the counts array if the end was truncated due to no voxels
    # in later classes being segmented.
    seg_counts = np.append(seg_counts, np.zeros(num_bins - len(seg_counts)))

    # Calculate the probabilities and return.
    prob_bins = seg_counts / voxel_counts
    if scale:
        prob_bins *= scale / np.amax(prob_bins)

    return bins, prob_bins


def probability_map(volume, bins, prob_bins):
    """Make a random training map from the likelihood that a voxel is a pec."""

    # Increase end of last bin so that the voxel of maximum intensity across
    # all vols used to create the map gets put into the final bin.
    bins[-1] += 1

    # Make a random vols to use for selecting the voxels to train on.
    random_volume = np.random.random(volume.shape)

    # Assign each voxel its bin number.
    voxel_bin_ind = np.digitize(volume.mri_data, bins)

    # Create a vols where each voxel is replaced by the probability that it
    # is segmented (based only on its intensity value).
    prob_map = prob_bins[voxel_bin_ind.flatten() - 1].reshape(volume.shape)

    # Threshold according to random volumes and voxel probabilities.
    prob_map[prob_map < random_volume] = 0

    return prob_map


def segmentation_map(volumes):
    """Create a training map for any segmented voxel of a list of vols."""

    # Count the number of times each voxel is segmented.
    counts = sum([volume.seg_data for volume in volumes])

    return counts.astype(np.bool)


def half_half_map(volume, max_points=None, margins=(0, 0, 0)):
    """Create training map with equal # segmented and non-segmented voxels."""

    # Define a function to take a random sample of point indices from an array,
    # out of those points where the array contains a certain value.
    def sample_indices_by_value(array, value, max_samples):
        indices = np.where(array == value)
        points = np.array(indices).T

        if len(points) < max_samples:
            return points
        else:
            return np.array(random.sample(points, max_samples)).T

    # Create slices to use for extracting the inner part of the volume.
    margined_slices = [slice(margin, max_size - margin)
                       for margin, max_size in zip(margins, volume.shape)]

    # Extract the inner part (within the margins).
    margined_data = volume.seg_data[margined_slices]

    # Count number of segmented voxels and non-segmented voxels.
    num_seg_points = np.count_nonzero(margined_data == 1)
    num_non_seg_points = np.count_nonzero(margined_data == 0)

    # Find the number of points to sample for each class.
    num_class_points = min(num_seg_points, num_non_seg_points)
    if max_points is not None:
        num_class_points = min(num_class_points, max_points // 2)

    # Return a blank map if either class is empty.
    if num_class_points == 0:
        return np.full(volume.shape, False, dtype='bool')

    # Generate the samples of points to use for each class.
    seg_indices = sample_indices_by_value(margined_data, 1, num_class_points)
    non_seg_indices = \
        sample_indices_by_value(margined_data, 0, num_class_points)

    # Add the margin offsets back on.
    for i, margin in enumerate(margins):
        seg_indices[i] += margin
        non_seg_indices[i] += margin

    # Create the map.
    half_half = np.full(volume.shape, False, dtype='bool')
    half_half[tuple(seg_indices)] = True
    half_half[tuple(non_seg_indices)] = True

    return half_half


def full_map(volume):
    """Create a training map encompassing the whole of the input vol."""

    return np.ones(volume.shape)
