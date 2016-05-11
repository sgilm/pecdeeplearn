from __future__ import division

import numpy as np


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

    # Threshold according to random vols and voxel probabilities.
    prob_map[prob_map < random_volume] = 0

    return prob_map


def segmentation_map(volumes):
    """Create a training map for any segmented voxel of a list of vols."""

    # Count the number of times each voxel is segmented.
    counts = sum([volume.seg_data for volume in volumes])

    return counts.astype(np.bool)


def half_half_map(volume):
    """Create training map with equal # segmented and non-segmented voxels."""

    # Count the number of segmented voxels.
    num_segs = np.sum(volume.seg_data)

    # Return a blank map if no voxels are segmented.
    if num_segs == 0:
        return np.zeros(volume.shape, dtype='bool')

    # Generate the same number of random points that are not segmented.
    non_seg_points = set()
    while len(non_seg_points) < num_segs:
        point = []
        for max_size in volume.shape:
            point.append(np.random.randint(0, max_size))
        point = tuple(point)
        if volume.seg_data[point] != 1:
            non_seg_points.add(point)

    # Create half-half map, starting with all segmented voxels.
    half_half = volume.seg_data.astype('bool')

    # Convert list of points into tuple of axis indices.
    non_seg_indices = tuple(zip(*non_seg_points))

    # Add to map.
    half_half[non_seg_indices] = True

    return half_half


def full_map(volume):
    """Create a training map encompassing the whole of the input vol."""

    return np.ones(volume.shape)
