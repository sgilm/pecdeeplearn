import numpy as np
import scipy.misc
import itertools


class Volume:
    """
    A class representing a scanned, three dimensional volume.

    Args:
        mri_data (numpy.memmap): a pointer to a set of mri data, with single
            voxel intensity values that range from 0 to 255.
        seg_data (numpy.memmap): a pointer to a set of segmentation data, with
            binary voxels.
        plane (str): the orientation of the data sets in their first indexing
            dimension.  E.g. if self.plane == 'axial', then self.mri_data[100]
            takes an axial slice.  Can be 'sagittal', 'axial', or 'coronal'.

    Attributes:
        mri_data (numpy.memmap): equals arg
        seg_data (numpy.memmap): equals arg
        plane (str): equals arg
        
    """
    
    def __init__(self, mri_data, seg_data, plane='sagittal'):
        self.mri_data = mri_data
        self.seg_data = seg_data

        # Check the first three dimensions are consistent.
        if mri_data.shape[:3] != seg_data.shape[:3]:
            raise Exception('Data dimensions are inconsistent.')

        self.shape = mri_data.shape[:3]

        # The default orientation for data from file is 'sagittal'.
        self.plane = plane

    def switch_plane(self, plane):
        """Switch the orientation of the data."""

        def swap_axes(volume, i, j):
            volume.mri_data = np.swapaxes(volume.mri_data, i, j)
            volume.seg_data = np.swapaxes(volume.seg_data, i, j)

        # Keep the planes to switch in an unordered collection.
        switch = {plane, self.plane}

        # Choose behaviour based on planes to switch.
        if switch == {'sagittal', 'coronal'}:
            swap_axes(self, 0, 1)
        elif switch == {'sagittal', 'axial'}:
            swap_axes(self, 0, 2)
        elif switch == {'axial', 'coronal'}:
            if plane == 'axial':
                swap_axes(self, 1, 2)
                swap_axes(self, 0, 1)
            else:
                swap_axes(self, 0, 1)
                swap_axes(self, 1, 2)

        self.plane = plane

    def get_slice(self, slice_index):
        """
        Get a data slice from the first dimension in the current orientation.

        Arg:
            slice_index (int): the index of the slice in the first dimension

        Returns:
            mri_slice_data (numpy.array): a two dimensional array of voxel
                intensities from the mri_data
            seg_slice_data (numpy.array): a two dimensional array of binary
                voxels defining a sgementation.
                
        """
        mri_slice_data = np.array(self.mri_data[slice_index])
        seg_slice_data = np.array(self.seg_data[slice_index])

        # Cleanup may be required to reshape the arrays.
        if len(mri_slice_data.shape) > 2:
            m, n = mri_slice_data.shape[:2]
            mri_slice_data = mri_slice_data.reshape(m, n)
            seg_slice_data = seg_slice_data.reshape(m, n)
            
        return mri_slice_data, seg_slice_data

    def show_slice(self, slice_index):
        """Show mri and seg data for a particular slice as a picture."""
        for slice_data in self.get_slice(slice_index):
            scipy.misc.toimage(slice_data).show()

    def __getitem__(self, indices):
        """Define volume slicing behaviour."""
        if not isinstance(indices, tuple):
            indices = [indices]
        good_indices = []
        for index in indices:
            if isinstance(index, int):
                good_indices.append(slice(index, index + 1))
            elif isinstance(index, slice):
                good_indices.append(index)
            else:
                raise Exception("Volume indices should be ints or slices.")

        return Volume(self.mri_data[good_indices],
                      self.seg_data[good_indices], plane=self.plane)


class BatchIterator:
    def __init__(self, volume):
        self.volume = volume
        self.features = []
        self.input_size = None

    def add_feature(self, feature):
        self.features.append(feature)
        self.input_size = None

    def get_input_size(self):
        if not self.input_size:

            # Get the size of the input vector.
            ranges_to_iterate = [range(size) for size in self.volume.shape]
            num_inputs = 0
            for point in itertools.product(*ranges_to_iterate):
                try:
                    num_inputs = 0
                    for feature in self.features:
                        feature_data = feature(self.volume, point)
                        num_inputs += feature_data.size
                    break
                except:
                    continue

            if num_inputs == 0:
                raise Exception

            self.input_size = num_inputs

        return self.input_size

    def iterate(self, batch_size):
        ranges_to_iterate = [range(size) for size in self.volume.shape]

        input_size = self.get_input_size()

        input_batch = np.empty((batch_size, input_size), dtype='float64')
        output_batch = np.empty(batch_size, dtype='float64')

        count = 0
        for point in itertools.product(*ranges_to_iterate):
            try:
                temp = []
                for feature in self.features:
                    temp.extend(list(feature(self.volume, point).flatten()))
                input_batch[count % batch_size] = np.array(temp)
                output_batch[count % batch_size] = self.volume.seg_data[point]
                count += 1
                if count % batch_size == 0:
                    yield input_batch, output_batch
            except:
                continue