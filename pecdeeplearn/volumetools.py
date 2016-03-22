import nibabel
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
    
    def __init__(self, mri_data, seg_data, plane='saggital'):
        self.mri_data = mri_data
        self.seg_data = seg_data

        # The default orientation for data from file is 'sagittal'.
        self.plane = plane

    def switch_plane(self, plane):
        """Switch the orientation of the data."""

        def swap_axes(vol, i, j):
            vol.mri_data = np.swapaxes(vol.mri_data, i, j)
            vol.seg_data = np.swapaxes(vol.seg_data, i, j)

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
        """Show mri and segmentation data for a particular slice as a picture."""
        for slice_data in self.get_slice(slice_index):
            scipy.misc.toimage(slice_data).show()

    def iterate_batches(self, kernel_shape, batch_size):
        """
        Generate training datasets that use a specified kernel shape.

        Args:
            kernel_shape (list): dimensions (in voxels) of the kernel to use for
                the generated training data
            batch_size (int): the number of voxels to include in each batch

        Yields:
            mri_batch (numpy.array): dimensions are batch_size x kernel_shape[0]
                x kernel_shape[1] x kernel_shape[2], containing training data
            seg_batch (numpy.array): dimensions are batch_size, containing
                binary voxels from the segmentation data
                
        """
        # Ensure the kernel dimensions are odd, so it can be centred on a voxel.
        for i in range(len(kernel_shape)):
            if kernel_shape[i] % 2 == 0:
                kernel_shape[i] -= 1

        # Extend the kernel shape if fewer dimensions are given.
        if len(kernel_shape) != 3:
            kernel_shape += [1] * (3 - len(kernel_shape))

        # Should be 3, since we're in 3 dimensions.
        num_dim = len(kernel_shape)

        # Create arrays of offsets defining the kernel about a centre voxel.
        kernel_starts = [-(kernel_size // 2) for kernel_size in kernel_shape]
        kernel_stops = [kernel_size // 2 + 1 for kernel_size in kernel_shape]
        kernel_ranges = [np.array([kernel_starts[i], kernel_stops[i]]) \
                for i in range(num_dim)]

        data_shape = self.mri_data.shape[:num_dim]

        def frame(n, kernel_size):
            start = kernel_size // 2
            stop = n - start
            return range(start, stop)

        # Create frames defining which points on the interior of the volume to process.
        frames = [frame(data_shape[i], kernel_shape[i]) for i in range(num_dim)]

        # Preallocate storage for numpy arrays.
        mri_batch = np.empty([batch_size] + kernel_shape, dtype='int32')
        seg_batch = np.empty(batch_size, dtype='int32')

        # Loop through each valid point and return batches when full.
        count = 0
        for point in itertools.product(*frames):
            index_ranges = [list(kernel_ranges[i] + point[i]) for i in range(num_dim)]
            index_slices = tuple([slice(*index_range) for index_range in index_ranges])
            mri_batch[count % batch_size] = np.reshape(self.mri_data[index_slices], kernel_shape)
            seg_batch[count % batch_size] = np.reshape(self.seg_data[point], 1)
            count += 1
            if count % batch_size == 0:
                yield mri_batch, seg_batch

    def __getitem__(self, indices):
        """Define volume slicing behaviour."""
        good_indices = []
        for index in indices:
            if isinstance(index, int):
                good_indices.append(slice(index, index + 1))
            else:
                good_indices.append(index)
        return Volume(self.mri_data[good_indices], self.seg_data[good_indices], plane=self.plane)


if __name__ == '__main__':

    mri = nibabel.load('../data/mris/VL00035.hdr')
    seg =  nibabel.load('../data/segmentations/segpec_VL00035.nii')
    vol = Volume(mri.get_data(), seg.get_data())
    vol.switch_plane('axial')
    small_vol = vol[100, :250, :250]
    kernel_shape = [1, 29, 29]
    small_vol.show_slice(0) 
