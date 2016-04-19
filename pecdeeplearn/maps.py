import numpy as np


def probability_bins(volumes, num_bins=25, scale=None):
    """
    Create intensity bins and probabilities a voxel of a bin is segmented.

    Args
        volumes (list): a list of Volume objects that are to be used for
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

    # Create a corresponding volume with each element being a bin label.
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
    # Increase end of last bin so that the voxel of maximum intensity across
    # all volumes used to create the map gets put into the final bin.
    bins[-1] += 1
    random_volume = np.random.random(volume.shape)
    voxel_bin_ind = np.digitize(volume.mri_data, bins)
    prob_map = prob_bins[voxel_bin_ind.flatten() - 1].reshape(volume.shape)
    prob_map[prob_map < random_volume] = 0

    return prob_map


def segmentation_map(volumes):
    counts = sum([volume.seg_data for volume in volumes])

    return counts.astype(np.bool)


def half_half_map(volume):
    """Create half-half training map."""

    num_segs = np.sum(volume.seg_data)
    non_seg_points = set()
    while len(non_seg_points) < num_segs:
        point = []
        for max_size in volume.shape:
            point.append(np.random.randint(0, max_size))
        point = tuple(point)
        if volume.seg_data[point] != 1:
            non_seg_points.add(point)

    half_half = volume.seg_data.astype('bool')

    non_seg_indices = tuple(zip(*non_seg_points))
    half_half[non_seg_indices] = True

    return half_half


if __name__ == '__main__':
    import volumetools as vt
    volume = vt.load_volume(vt.list_volumes()[0])
    volume.switch_plane('axial')

    half_half = half_half_map(volume)

    import scipy.misc
    scipy.misc.toimage(half_half[120]).show()