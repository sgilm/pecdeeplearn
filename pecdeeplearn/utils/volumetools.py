import os
import pickle
import nibabel
import numpy as np
import copy

from . import general
from ..extraction import Volume


def list_volumes():
    """List the names of available volumes that have the required data."""

    # Get the path to the data.
    data_path = general.datapath()

    # Loop through mri files and find .hdr and .img files.
    hdr_volumes = []
    img_volumes = []
    for filename in os.listdir(os.path.join(data_path, 'mris')):
        if '.hdr' in filename:
            hdr_volumes.append(filename.split('.')[0])
        elif '.img' in filename:
            img_volumes.append(filename.split('.')[0])

    # Valid volumes must have both .hdr and .img files.
    mri_volumes = [volume for volume in hdr_volumes if volume in img_volumes]

    # Loop through and list volumes with segmentation files.
    seg_volumes = []
    for filename in os.listdir(os.path.join(data_path, 'segmentations')):
        volume = filename.split('.')[0].replace('segpec_', '')
        seg_volumes.append(volume)

    # Loop through and list volumes with landmarks.
    landmark_volumes = []
    for volume in next(os.walk(os.path.join(data_path, 'landmarks')))[1]:
        landmark_volumes.append(volume)

    # Valid volumes must have mri, segmentation and landmark data.
    volumes = [volume for volume in seg_volumes
               if volume in mri_volumes and volume in landmark_volumes]

    return volumes


def load_volume(volume_name):
    """Load a volumes of a specified name."""

    # Get the path to the data.
    data_path = general.datapath()

    # Form the filenames for mri and segmentation data.
    mri_filename = volume_name + '.hdr'
    seg_filename = 'segpec_' + volume_name + '.nii'

    # Load mri and segmentation data.
    mri = nibabel.load(os.path.join(data_path, 'mris', mri_filename))
    seg = nibabel.load(os.path.join(data_path, 'segmentations', seg_filename))

    # Retrieve data.
    volume_data = [mri.get_data(), seg.get_data()]

    # Swap axes of data into the best orientation for viewing
    # ([axial][coronal][sagittal]).
    for i in range(len(volume_data)):
        volume_data[i] = np.swapaxes(volume_data[i], 0, 2)

    # Loop through landmarks to build a dictionary.
    landmarks = {}
    landmark_path = os.path.join(data_path, 'landmarks', volume_name)
    for filename in os.listdir(landmark_path):
        with open(os.path.join(landmark_path, filename), 'rb') as f:

            # Unpickle the pickled data.
            landmark_dict = pickle.load(f, encoding='latin1')

            # Strip the relevant data.
            name = landmark_dict['name']
            data = landmark_dict['data']['default']

            # Swap coordinates to be consistent with the axial orientation
            # of the mri and segmentation data.
            data[0], data[1] = data[1], data[0]
            data[0], data[2] = data[2], data[0]

            # Add to dictionary.
            landmarks[name] = data

    # Create the volumes.
    volume = Volume(volume_data[0],
                    volume_data[1],
                    copy.deepcopy(landmarks),
                    'acs')

    # Mirror the volumes in the sagittal plane and reassign landmarks so that
    # it is consistent.
    volume.mirror('s')
    volume.landmarks = landmarks

    # One final mirror to put the data in the most natural shape.
    volume.mirror('a')

    return volume


def pickle_volume(volume, filename):
    full_filename = os.path.join(general.datapath(), 'volumes', filename)
    general.pickle_object(volume, full_filename)


def unpickle_volume(filename):
    full_filename = os.path.join(general.datapath(), 'volumes', filename)
    return general.unpickle_object(full_filename)
