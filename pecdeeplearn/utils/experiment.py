from __future__ import division

import os
import pickle
import nibabel
import numpy as np
import copy

from ..extraction import Volume


class Experiment:
    """A class to record parameters and results, or load previous results."""

    def __init__(self, data_path, name):

        # Record the experiment identification information.
        self.data_path = data_path
        self.name = name

        # Create paths for the data subfolders.
        self.results_path = os.path.join(self.data_path, 'results', self.name)
        self.mris_path = os.path.join(self.data_path, 'mris')
        self.segs_path = os.path.join(self.data_path, 'segmentations')
        self.landmarks_path = os.path.join(self.data_path, 'landmarks')

        # Initialise a dictionary for holding experiment parameters.
        self.params = {}

    def _update_results_path(self):
        """Updates the name of the current experiment to avoid overwriting."""

        # While the current name is taken, increment a count tag at the end of
        # the name.
        count = 1
        while os.path.isdir(self.results_path):
            self.results_path = os.path.join(self.data_path,
                                             'results',
                                             self.name + str(count))
            count += 1

        # Make the new directory.
        os.mkdir(self.results_path)

    def add_param(self, key, value):
        self.params[key] = value

    def list_volumes(self):
        """List the names of available vols that have the required data."""

        # Loop through mri files and find .hdr and .img files.
        hdr_volumes = []
        img_volumes = []
        for name in os.listdir(self.mris_path):
            if '.hdr' in name:
                hdr_volumes.append(name.split('.')[0])
            elif '.img' in name:
                img_volumes.append(name.split('.')[0])

        # Valid vols must have both .hdr and .img files.
        mri_volumes = [volume for volume in hdr_volumes if
                       volume in img_volumes]

        # Loop through and list vols with segmentation files.
        seg_volumes = []
        for name in os.listdir(self.segs_path):
            volume = name.split('.')[0].replace('segpec_', '')
            seg_volumes.append(volume)

        # Loop through and list vols with landmarks.
        landmark_volumes = []
        for volume in next(os.walk(self.landmarks_path))[1]:
            landmark_volumes.append(volume)

        # Valid vols must have mri, segmentation and landmark data.
        volumes = [volume for volume in seg_volumes
                   if volume in mri_volumes and volume in landmark_volumes]

        return volumes

    def load_volume(self, volume_name):
        """Load a volume of a specified name."""

        # Form the filenames for mri and segmentation data.
        mri_filename = volume_name + '.hdr'
        seg_filename = 'segpec_' + volume_name + '.nii'

        # Load mri and segmentation data.
        mri = nibabel.load(os.path.join(self.mris_path, mri_filename))
        seg = nibabel.load(os.path.join(self.segs_path, seg_filename))

        # Retrieve data.
        volume_data = [mri.get_data(), seg.get_data()]

        # Swap axes of data into the best orientation for viewing
        # ([axial][coronal][sagittal]).
        for i in range(len(volume_data)):
            volume_data[i] = np.swapaxes(volume_data[i], 0, 2)

        # Loop through landmarks to build a dictionary.
        landmarks = {}
        landmark_path = os.path.join(self.landmarks_path, volume_name)
        for filename in os.listdir(landmark_path):
            with open(os.path.join(landmark_path, filename), 'rb') as f:

                # Unpickle the pickled data.
                # landmark_dict = pickle.load(f, encoding='latin1') for Python3
                landmark_dict = pickle.load(f)

                # Strip the relevant data.
                name = landmark_dict['name']
                data = landmark_dict['data']['default']

                # Swap coordinates to be consistent with the axial orientation
                # of the mri and segmentation data.
                data[0], data[1] = data[1], data[0]
                data[0], data[2] = data[2], data[0]

                # Add to dictionary.
                landmarks[name] = data

        # Create the vols.
        volume = Volume(volume_data[0],
                        volume_data[1],
                        copy.deepcopy(landmarks),
                        'acs')

        # Mirror the vols in the sagittal plane and reassign landmarks so that
        # it is consistent.
        volume.mirror('s')
        volume.landmarks = landmarks

        # One final mirror to put the data in the most natural shape.
        volume.mirror('a')

        return volume

    def pickle_volume(self, volume, name):
        """Pickle a (usually predicted) volume into the results directory."""

        # Make sure another experiment isn't being overwritten.
        self._update_results_path()

        # Save the volume.
        with open(os.path.join(self.results_path, name), 'wb') as f:
            pickle.dump(volume, f, -1)

    def unpickle_volume(self, name):
        """Unpickle a volume from the current results directory."""

        with open(os.path.join(self.results_path, name), 'rb') as f:
            volume = pickle.load(f)
        return volume

    def save_network(self, net, name):
        """Save a network's weights into the results directory."""

        # Make sure another experiment isn't being overwritten, and save.
        self._update_results_path()
        net.save_params_to(os.path.join(self.results_path, name))

    def load_network(self, net, name):
        """Load a network's weights and initialise it."""

        net.load_params_from(os.path.join(self.results_path, name))
        net.initialize()

    def record_params(self):
        """Record the current experiment's parameters."""

        # Make sure another experiment isn't being overwritten.
        self._update_results_path()

        # Write the params dictionary.
        with open(os.path.join(self.results_path, 'params.txt'), 'w') as f:
            for key, value in self.params.items():
                f.write('{} = {}\n'.format(key, value))