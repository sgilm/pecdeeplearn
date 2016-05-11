from __future__ import division

import numpy as np

def standardise_volumes(volumes):
    """Creates a standardised dataset."""

    all_mri_data = [volume.mri_data for volume in volumes]

    overall_mean = np.mean(all_mri_data)
    overall_std = np.std(all_mri_data)

    for volume in volumes:
        volume.mri_data = volume.mri_data - overall_mean
        volume.mri_data = volume.mri_data / overall_std