from __future__ import division

import numpy as np
import copy
import matplotlib.pylab as plt


class Volume:
    """
    A class representing a scanned, three dimensional volume.

    Args
        name (str): the name of the volume (usually VL.....).
        header (nibabel.Nifti1Header): contains metadata for the volume.
        affine (numpy.ndarray): maps voxel indices to a spatial location.
        mri_data (numpy.memmap): a pointer to a set of mri data, with single
            voxel intensity values.
        seg_data (numpy.memmap): a pointer to a set of segmentation data, with
            binary voxels.
        landmarks (dict): contains landmark names as keys and 3 element
            coordinate arrays as values.  The coordinates are spatial (not
            indices), so the data contained in the volume header has to be used
            to interpret them.

    Attributes
        name (str): equals arg.
        header (nibabel.Nifti1Header): equals arg.
        affine (numpy.ndarray): equals arg.
        mri_data (numpy.memmap): equals arg.
        seg_data (numpy.memmap): equals arg.
        landmarks (dict): equals arg.
        orientation (str): equals arg.
        shape (tuple): gives the dimensions of the volume.  Should be
            consistent between mri and seg data.

    """

    def __init__(self, name, header, affine, mri_data, seg_data, landmarks):

        # Record volume name.
        self.name = name

        # Record volume metadata.
        self.header = header
        self.affine = affine

        # Squeeze to clean up data with unwanted singleton dimensions.
        mri_dims_to_squeeze = tuple(range(3, len(mri_data.shape)))
        seg_dims_to_squeeze = tuple(range(3, len(seg_data.shape)))
        self.mri_data = np.squeeze(mri_data, axis=mri_dims_to_squeeze)
        self.seg_data = np.squeeze(seg_data, axis=seg_dims_to_squeeze)

        # Record landmarks.
        self.landmarks = landmarks

        # Check the dimensions are consistent.
        if self.mri_data.shape != self.seg_data.shape:
            raise Exception('Data dimensions are inconsistent.')

        # If they are consistent, set this as the shape of the volume.
        self.shape = self.mri_data.shape

    def get_slice(self, slice_index, axis):
        """
        Get a slice of data along a specified axis.

        Arg:
            slice_index (int): the index of the slice to extract.
            axis (int): the axis from which to extract the slice.

        Returns:
            mri_slice_data (numpy.array): a two dimensional array of voxel
                intensities from the mri data.
            seg_slice_data (numpy.array): a two dimensional array of binary
                voxels from the seg data.

        """
        mri_slice_data = self.mri_data.take(slice_index, axis=axis)
        seg_slice_data = self.seg_data.take(slice_index, axis=axis)

        return mri_slice_data, seg_slice_data

    def show_slice(self, slice_index, axis, include_seg=True, seg_cmap='Reds',
                   num_rotations=0):
        """Show mri and seg data for a particular slice as a picture."""

        # Get the data for both mri and seg.
        mri_slice_data, seg_slice_data = self.get_slice(slice_index, axis)
        mri_slice_data = np.rot90(mri_slice_data, k=num_rotations)
        seg_slice_data = np.rot90(seg_slice_data, k=num_rotations)

        # Plot the mri image in greyscale.
        plt.figure(frameon=False)
        plt.axis('off')
        plt.imshow(mri_slice_data, cmap='gray')

        # Show the segmentation (with transparency) if required.
        if include_seg:
            masked_seg = np.ma.masked_equal(seg_slice_data, 0)
            plt.imshow(masked_seg, cmap=seg_cmap, alpha=0.4, vmin=0, vmax=1)

        # Show the plot.
        plt.show()

    def bounding_box(self, margins=None):
        """Find the indices of the segmentation's bounding box."""

        # Get the indices of the points defining the bounding box.
        seg_indices = np.nonzero(self.seg_data)
        min_bounding_indices = np.min(seg_indices, axis=1)
        max_bounding_indices = np.max(seg_indices, axis=1)

        # Apply the margins to the bounding box.
        if margins is not None:
            for i, margin in enumerate(margins):
                min_bounding_indices[i] -= margin
                min_bounding_indices[i] = max(min_bounding_indices[i], 0)
                max_bounding_indices[i] += margin
                max_bounding_indices[i] = min(max_bounding_indices[i],
                                              self.shape[i] - 1)

        return min_bounding_indices, max_bounding_indices

    def __getitem__(self, indices):
        """
        Define volume slicing behaviour.

        (It is better not to use this if possible, because some of the metadata
        associated with the volume becomes invalid.)

        """

        # If the indices are not a valid iterable, put them in a list for
        # processing.
        try:
            _ = (i for i in indices)
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
                new_origin.append(index.start or 0)

            # Other types are invalid.
            else:
                raise Exception("Volume indices should be integers or slices.")

        # Fill in the origin of unused index dimensions with zeroes.
        new_origin.extend([0] * (3 - len(new_origin)))

        # Shift the landmarks for the new volume based on the new origin.
        new_landmarks = copy.deepcopy(self.landmarks)
        new_origin_array = np.array(new_origin)
        spacing_array = np.array(self.header.get_zooms()[:3])
        for landmark_location in new_landmarks.values():
            landmark_location -= new_origin_array * spacing_array

        return Volume(self.name,
                      self.header,
                      self.affine,
                      self.mri_data[processed_indices],
                      self.seg_data[processed_indices],
                      new_landmarks
                      )
