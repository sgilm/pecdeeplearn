from __future__ import division

import os
import cPickle as pickle
import nibabel
import sys

from ..extraction import Volume


class Experiment:
    """A class to record parameters and results, or load previous results."""

    def __init__(self, data_path):

        # Record main data path information.
        self.data_path = data_path

        # Record paths for subfolders.
        self.results_path = os.path.join(self.data_path, 'results')
        self.mris_path = os.path.join(self.data_path, 'mris')
        self.segs_path = os.path.join(self.data_path, 'segmentations')
        self.landmarks_path = os.path.join(self.data_path, 'landmarks')

        # Initialise path to this exact experiment.
        self.experiment_path = None

        # Initialise dictionaries for holding parameters and results.
        self.params = {}
        self.results = {}

    def create_experiment(self, name):
        """Create new directory for an experiment and initialise the class."""

        # Add an index to the experiment name, and advance it until it is the
        # latest index in the directory.
        index = 1
        indexed_name = name + "_" + str(index)
        while indexed_name in os.listdir(self.results_path):
            index += 1
            indexed_name = name + "_" + str(index)

        # Save the path to this experiment, and create a new directory.
        self.experiment_path = os.path.join(self.results_path, indexed_name)
        os.mkdir(self.experiment_path)

    def load_experiment(self, name):
        """Initialise the experiment to point to an existing one."""

        # Make sure the experiment exists before forming the experiment path.
        if not os.path.isdir(os.path.join(self.results_path, name)):
            raise Exception('Experiment does not exist.')
        else:
            self.experiment_path = os.path.join(self.results_path, name)

    def add_param(self, key, value):
        self.params[key] = value

    def add_result(self, key, value):
        self.results[key] = value

    def list_volumes(self):
        """List the names of available volumes that have the required data."""

        # Loop through mri files and find .hdr and .img files.
        hdr_volumes = []
        img_volumes = []
        for name in os.listdir(self.mris_path):
            if '.hdr' in name:
                hdr_volumes.append(name.split('.')[0])
            elif '.img' in name:
                img_volumes.append(name.split('.')[0])

        # Valid volumes must have both .hdr and .img files.
        mri_volumes = [volume for volume in hdr_volumes if
                       volume in img_volumes]

        # Loop through and list volumes with segmentation files.
        seg_volumes = []
        for name in os.listdir(self.segs_path):
            volume = name.split('.')[0].replace('segpec_', '')
            seg_volumes.append(volume)

        # Loop through and list volumes with landmarks.
        landmark_volumes = []
        for volume in next(os.walk(self.landmarks_path))[1]:
            landmark_volumes.append(volume)

        # Valid volumes must have mri, segmentation and landmark data.
        volumes = [volume for volume in seg_volumes
                   if volume in mri_volumes and volume in landmark_volumes]

        return sorted(volumes)

    def load_volume(self, volume_name, experiment=False, suffix=''):
        """Load a volume with landmark, header, and affine metadata."""

        # Load the mri data, which always comes from the same directory.
        mri_filename = volume_name + '.hdr'
        mri = nibabel.load(os.path.join(self.mris_path, mri_filename))

        # Load the segmentation data from a different location depending on the
        # input argument.
        if experiment:
            seg_filename = volume_name + suffix + '_seg.nii'
            seg = nibabel.load(os.path.join(self.experiment_path,
                                            seg_filename))
        else:
            seg_filename = 'segpec_' + volume_name + '.nii'
            seg = nibabel.load(os.path.join(self.segs_path, seg_filename))

        # Loop through landmarks to build a dictionary.
        landmarks = {}
        vol_landmarks_path = os.path.join(self.landmarks_path, volume_name)
        for filename in os.listdir(vol_landmarks_path):
            with open(os.path.join(vol_landmarks_path, filename), 'rb') as f:

                # Unpickle the pickled data.
                # landmark_dict = pickle.load(f, encoding='latin1') for Python3
                landmark_dict = pickle.load(f)

                # Strip the relevant data.
                name = landmark_dict['name']
                data = landmark_dict['data']['default']

                # Alter coordinates to be consistent with the orientation of
                # the mri and segmentation data.
                spacing = mri.get_header().get_zooms()[0]
                size = mri.get_data().shape[0]
                data[0], data[1] = spacing * size - data[1], data[0]

                # Add to dictionary.
                landmarks[name] = data

        # Create the volume.
        volume = Volume(
            volume_name,
            mri.get_header(),
            mri.get_affine(),
            mri.get_data(),
            seg.get_data(),
            landmarks
        )

        return volume

    def pickle_volume(self, volume):
        """Pickle a (usually predicted) volume into the results directory."""

        with open(os.path.join(self.experiment_path, volume.name), 'wb') as f:
            pickle.dump(volume, f, -1)

    def unpickle_volume(self, name):
        """Unpickle a volume from the current results directory."""

        with open(os.path.join(self.experiment_path, name), 'rb') as f:
            volume = pickle.load(f)
        return volume

    def pickle_network(self, net, name):
        """Pickle a network into the results directory."""

        # Increase the default recursion limit to allow pickling.
        sys.setrecursionlimit(10000)

        # Pickle the network.
        with open(os.path.join(self.experiment_path, name), 'wb') as f:
            pickle.dump(net, f, -1)

    def unpickle_network(self, name):
        """Unpickle a network from the results directory."""

        # Unpickle the network.
        with open(os.path.join(self.experiment_path, name), 'rb') as f:
            network = pickle.load(f)
        return network

    def save_network_weights(self, net, name):
        """Save a network's weights into the results directory."""

        net.save_params_to(os.path.join(self.experiment_path, name))

    def load_network_weights(self, net, name):
        """Load a network's weights and initialise it."""

        net.load_params_from(os.path.join(self.experiment_path, name))
        net.initialize()

    def record(self):
        """Record the current experiment's parameters and results."""

        # Write the params and results dictionaries.
        labels = ['params', 'results']
        datasets = [self.params, self.results]
        for label, dataset in zip(labels, datasets):
            with open(os.path.join(self.experiment_path, label + '.txt'),
                      'w') as f:
                for key, value in dataset.items():
                    f.write('{} = {}\n'.format(key, value))

    def export_nii(self, volume, mri=False, seg=True):
        """Export a .nii file from an instance of the Volume class."""

        # Export the mri data.
        if mri:
            mri_img = nibabel.Nifti1Image(volume.mri_data, volume.affine,
                                          volume.header)
            nibabel.save(mri_img, os.path.join(self.experiment_path,
                                               volume.name + '_mri.nii'))

        # Export the segmentation data.
        if seg:
            seg_img = nibabel.Nifti1Image(volume.seg_data, volume.affine,
                                          volume.header)
            nibabel.save(seg_img, os.path.join(self.experiment_path,
                                               volume.name + '_seg.nii'))
