from __future__ import division

import numpy as np
import copy
import matplotlib.pylab as plt


class Volume:
    """
    A class representing a scanned, three dimensional vols.

    Args
        mri_data (numpy.memmap): a pointer to a set of mri data, with single
            voxel intensity values.
        seg_data (numpy.memmap): a pointer to a set of segmentation data, with
            binary voxels.
        landmarks (dict): contains landmark names as keys and 3 element
            coordinate arrays as values.
        orientation (str): a 3 element string containing the chars a, c and s
            which gives the order of the axial, coronal and sagittal axes in
            the data arrays.

    Attributes
        mri_data (numpy.memmap): equals arg.
        seg_data (numpy.memmap): equals arg.
        landmarks (dict): equals arg.
        orientation (str): equals arg.
        shape (tuple): gives the dimensions of the vols.  Should be
            consistent between mri and seg data

    """

    def __init__(self, mri_data, seg_data, landmarks, orientation):

        # Squeeze to clean up data with unwanted singleton dimensions.
        mri_dims_to_squeeze = tuple(range(3, len(mri_data.shape)))
        seg_dims_to_squeeze = tuple(range(3, len(seg_data.shape)))
        self.mri_data = np.squeeze(mri_data, axis=mri_dims_to_squeeze)
        self.seg_data = np.squeeze(seg_data, axis=seg_dims_to_squeeze)

        # Record landmarks and orientation.
        self.landmarks = landmarks
        self.orientation = orientation

        # Check the dimensions are consistent.
        if self.mri_data.shape != self.seg_data.shape:
            raise Exception('Data dimensions are inconsistent.')

        # If they are consistent, set this as the shape of the vols.
        self.shape = self.mri_data.shape

    def _swap_axes(self, i, j):
        """Swaps the axes of all data in the vols."""

        # Treat each piece of data separately.
        self.mri_data = np.swapaxes(self.mri_data, i, j)
        self.seg_data = np.swapaxes(self.seg_data, i, j)
        for point in self.landmarks.values():
            point[i], point[j] = point[j], point[i]

    def switch_orientation(self, new_orientation):
        """
        Switch the orientation of the data.

        Args
            new_orientation (str): a 3 element string giving a new order for
                the axial, sagittal and coronal axes.

        """

        # Check the input is valid.
        if len(new_orientation) != 3:
            raise Exception('Input should be 3 element string of a, c and s.')

        # Create a list for working and loop through new orientation.
        updated = list(self.orientation)
        for i, plane in enumerate(new_orientation):

            # Get the index of the axes to swap with.
            j = updated.index(plane)

            # Swap the data and working orientation.
            self._swap_axes(i, j)
            updated[i], updated[j] = updated[j], updated[i]

        # Set the new orientation and shape.
        self.orientation = new_orientation
        self.shape = self.mri_data.shape

    def get_slice(self, slice_index):
        """
        Get a data slice from the first dimension in the current orientation.

        Arg:
            slice_index (int): the index of the slice in the first dimension.

        Returns:
            mri_slice_data (numpy.array): a two dimensional array of voxel
                intensities from the mri_data.
            seg_slice_data (numpy.array): a two dimensional array of binary
                voxels defining a segmentation.

        """
        mri_slice_data = np.array(self.mri_data[slice_index])
        seg_slice_data = np.array(self.seg_data[slice_index])

        return mri_slice_data, seg_slice_data

    def show_slice(self, slice_index, include_seg=True, seg_cmap='Reds'):
        """Show mri and seg data for a particular slice as a picture."""

        # Get the data for both mri and seg.
        mri_slice_data, seg_slice_data = self.get_slice(slice_index)

        # Plot the mri image in greyscale.
        plt.figure()
        plt.imshow(mri_slice_data, cmap='gray')

        # Show the segmentation (with transparency) if required.
        if include_seg:
            masked_seg = np.ma.masked_equal(seg_slice_data, 0)
            plt.imshow(masked_seg, cmap=seg_cmap, alpha=0.4, vmin=0, vmax=1)

        # Show the plot.
        plt.show()

    def mirror(self, mirror_planes):
        """Reflect a vols in the specified planes.

        Args
            mirror_planes (str): contains any of the characters a, c, and s -
                which specify the planes to mirror.  For example,
                mirror_planes = 'as' will mirror the vols in the axial and
                sagittal planes, but not the coronal plane.

        """

        # Loop through each axis.
        for i, plane in enumerate(self.orientation):
            if plane in mirror_planes:

                # Swap data so the axis to be reversed is the first dimension.
                # Then reverse it and swap back.
                self._swap_axes(0, i)
                self.mri_data = self.mri_data[::-1]
                self.seg_data = self.seg_data[::-1]
                self._swap_axes(0, i)

                # Update the landmark coordinates.
                for point in self.landmarks.values():
                    point[i] = (self.shape[i] - 1) - point[i]

    def __getitem__(self, indices):
        """Define vols slicing behaviour."""

        # If the indices are not a valid iterable, put them in a list for
        # processing.
        try:
            _ = (e for e in indices)
        except TypeError:
            indices = [indices]

        # Check for too many dimensions.
        if len(indices) > 3:
            Exception('Too many index dimensions specified.')

        # Loop through to create a list of slices that works with numpy.memmap,
        # and get the coordinates of the new origin.
        processed_indices = []
        new_origin = []
        for index in indices:

            # Create a slice from an integer.
            if isinstance(index, int):
                processed_indices.append([index])
                new_origin.append(index)

            # If the index is already a slice, extract the starting element for
            # identifying the new origin.
            elif isinstance(index, slice):
                processed_indices.append(index)
                new_origin.append(index.start)

            # Other types are invalid.
            else:
                raise Exception("Volume indices should be integers or slices.")

        # Fill in the origin of unused index dimensions with zeroes.
        new_origin.extend([0] * (3 - len(new_origin)))

        # Shift the landmarks for the new vols based on the new origin.
        new_landmarks = copy.deepcopy(self.landmarks)
        new_origin_array = np.array(new_origin)
        for landmark_location in new_landmarks.values():
            landmark_location -= new_origin_array

        return Volume(self.mri_data[processed_indices],
                      self.seg_data[processed_indices],
                      new_landmarks,
                      self.orientation)