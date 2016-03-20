import nibabel
import numpy as np
import scipy.misc
import itertools


class Volume:
    def __init__(self, mri_data, seg_data):
        self.mri_data = mri_data
        self.seg_data = seg_data
        self.plane = 'sagittal'

    def switch_plane(self, plane):
        switch = {plane, self.plane}

        def swap_axes(vol, i, j):
            vol.mri_data = np.swapaxes(vol.mri_data, i, j)
            vol.seg_data = np.swapaxes(vol.seg_data, i, j)

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
        mri_slice_data = np.array(self.mri_data[slice_index])
        seg_slice_data = np.array(self.seg_data[slice_index])
        if len(mri_slice_data.shape) > 2:
            m, n = mri_slice_data.shape[:2]
            mri_slice_data = mri_slice_data.reshape(m, n)
            seg_slice_data = seg_slice_data.reshape(m, n)
        return mri_slice_data, seg_slice_data

    def show_slice(self, slice_index):
        for slice_data in self.get_slice(slice_index):
            scipy.misc.toimage(slice_data).show()

    def iterate_batches(self, kernel_shape, batch_size):
        for i in range(len(kernel_shape)):
            if kernel_shape[i] % 2 == 0:
                kernel_shape[i] -= 1

        if len(kernel_shape) != 3:
            kernel_shape += [1] * (3 - len(kernel_shape))

        num_dim = len(kernel_shape)

        kernel_starts = [-(kernel_size // 2) for kernel_size in kernel_shape]
        kernel_stops = [kernel_size // 2 + 1 for kernel_size in kernel_shape]

        kernel_ranges = [np.array([kernel_starts[i], kernel_stops[i]]) \
                for i in range(num_dim)]

        data_shape = self.mri_data.shape[:num_dim]

        def frame(n, kernel_size):
            start = kernel_size // 2
            stop = n - start
            return range(start, stop)

        frames = [frame(data_shape[i], kernel_shape[i]) for i in range(num_dim)]

        mri_batch = np.empty([batch_size] + kernel_shape, dtype='int32')
        seg_batch = np.empty(batch_size, dtype='int32')

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
        good_indices = []
        for index in indices:
            if isinstance(index, int):
                good_indices.append(slice(index, index + 1))
            else:
                good_indices.append(index)
        return Volume(self.mri_data[good_indices], self.seg_data[good_indices])


if __name__ == '__main__':

    mri = nibabel.load('../data/mris/VL00035.hdr')
    seg =  nibabel.load('../data/segmentations/segpec_VL00035.nii')
    vol = Volume(mri.get_data(), seg.get_data())
    vol.switch_plane('axial')
    small_vol = vol[100, :250, :250]
    kernel_shape = [1, 29, 29]
    small_vol.show_slice(0) 
