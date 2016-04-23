import os
import pickle
import nibabel
import numpy as np
import scipy.misc
import itertools
import datapath
import copy


def list_volumes():
    """List the names of available volumes that have the required data."""

    # Get the path to the data.
    data_path = datapath.get()

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
    land_volumes = []
    for volume in next(os.walk(os.path.join(data_path, 'landmarks')))[1]:
        land_volumes.append(volume)

    # Valid volumes must have mri, segmentation and landmark data.
    volumes = [volume for volume in seg_volumes
               if volume in mri_volumes and volume in land_volumes]

    return volumes


def load_volume(volume_name):
    """Load a volume of a specified name."""

    # Get the path to the data.
    data_path = datapath.get()

    # Form the filenames for mri and segmentation data.
    mri_filename = volume_name + '.hdr'
    seg_filename = 'segpec_' + volume_name + '.nii'

    # Load mri and segmentation data.
    mri = nibabel.load(os.path.join(data_path, 'mris', mri_filename))
    seg = nibabel.load(os.path.join(data_path, 'segmentations', seg_filename))

    # Loop through landmarks to build and dictionary.
    landmarks = {}
    landmark_path = os.path.join(data_path, 'landmarks', volume_name)
    for filename in os.listdir(landmark_path):
        with open(os.path.join(landmark_path, filename), 'rb') as f:

            # Unpickle the pickled data.
            landmark_dict = pickle.load(f, encoding='latin1')

            # Strip the relevant data.
            name = landmark_dict['name']
            data = landmark_dict['data']['default']

            # Swap coordinates to be consistent with the sagittal orientation
            # of the mri and segmentation data.
            data[0], data[1] = data[1], data[0]

            # Add to dictionary.
            landmarks[name] = data

    return Volume(mri.get_data(), seg.get_data(), landmarks)


def pickle_volume(volume, filename):
    full_filename = os.path.join(datapath.get(), 'saved', filename + '.pkl')
    with open(full_filename, 'wb') as f:
        pickle.dump(volume, f, -1)


def unpickle_volume(filename):
    full_filename = os.path.join(datapath.get(), 'saved', filename + '.pkl')
    with open(full_filename, 'rb') as f:
        volume = pickle.load(f)
    return volume


class Volume:
    """
    A class representing a scanned, three dimensional volume.

    Args
        mri_data (numpy.memmap): a pointer to a set of mri data, with single
            voxel intensity values that range from 0 to 255.
        seg_data (numpy.memmap): a pointer to a set of segmentation data, with
            binary voxels.
        landmarks (dict): contains landmark names as keys and 3 element
            coordinate arrays as values.

    Attributes
        mri_data (numpy.memmap): equals arg
        seg_data (numpy.memmap): equals arg
        landmarks (dict): equals arg
        shape (tuple): gives the dimensions of the volume arrays
        plane (str): the orientation of the data sets in their first indexing
            dimension.  E.g. if self.plane == 'axial', then self.mri_data[100]
            takes an axial slice.  Can be 'sagittal', 'axial', or 'coronal'.
        
    """
    
    def __init__(self, mri_data, seg_data, landmarks, plane='sagittal'):

        # Squeeze to clean up data with unwanted singleton dimensions.
        mri_dims_to_squeeze = tuple(range(3, len(mri_data.shape)))
        seg_dims_to_squeeze = tuple(range(3, len(seg_data.shape)))
        self.mri_data = np.squeeze(mri_data, axis=mri_dims_to_squeeze)
        self.seg_data = np.squeeze(seg_data, axis=seg_dims_to_squeeze)

        # Standardise the mri data.
        self.mri_data = self.mri_data - np.mean(self.mri_data)
        self.mri_data /= np.std(self.mri_data)

        # Record landmarks.
        self.landmarks = landmarks

        # Check the dimensions are consistent.
        if self.mri_data.shape != self.seg_data.shape:
            raise Exception('Data dimensions are inconsistent.')

        # If they are, set this as the shape of the volume.
        self.shape = self.mri_data.shape

        # The default orientation when data is read from file is 'sagittal'.
        self.plane = plane

    def switch_plane(self, plane):
        """Switch the orientation of the data."""

        # Define a function to swap the axes of all data in the Volume.
        def swap_axes(volume, i, j):
            volume.mri_data = np.swapaxes(volume.mri_data, i, j)
            volume.seg_data = np.swapaxes(volume.seg_data, i, j)
            for point in volume.landmarks.values():
                point[i], point[j] = point[j], point[i]

        # Keep the planes to switch in an unordered collection.
        switch = {plane, self.plane}

        # Choose behaviour based on the planes to switch between.
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

        # Update the shape and plane of the volume.
        self.shape = self.mri_data.shape
        self.plane = plane

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

    def show_slice(self, slice_index, slice_type='both'):
        """Show mri and seg data for a particular slice as a picture."""

        # Get the data for both mri and seg.
        mri_slice_data, seg_slice_data = self.get_slice(slice_index)

        # Set the image data according to the specified slice type to show.
        if slice_type == 'mri':
            image_data = mri_slice_data
        elif slice_type == 'seg':
            image_data = mri_slice_data
        elif slice_type == 'both':

            # Set the intensity of the segmented voxels to be the maximum
            # intensity of the mri (scipy.misc.toimage displays the maximum
            # intensity as white and 0 as black).
            seg_indices = np.nonzero(seg_slice_data)
            mri_slice_data[seg_indices] = np.amax(mri_slice_data)
            image_data = mri_slice_data
        else:
            raise Exception('Unrecognised slice type.')

        # Display the slice.
        scipy.misc.toimage(image_data).show()

    def __getitem__(self, indices):
        """Define volume slicing behaviour."""

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

        # Shift the landmarks for the new volume based on the new origin.
        new_landmarks = copy.deepcopy(self.landmarks)
        new_origin_array = np.array(new_origin)
        for landmark_location in new_landmarks.values():
            landmark_location -= new_origin_array

        return Volume(self.mri_data[processed_indices],
                      self.seg_data[processed_indices],
                      new_landmarks,
                      plane=self.plane)


class Extractor:
    """
    A class to iterate over a volume and extract data for use in a neural net.

    Attributes
        features (dict): a dictionary with keys as feature names, and values as
            functions that extract data corresponding to that feature from
            a volume.
        feature_sizes (dict): a dictionary with keys as feature names, and
            values tuples that give the dimensions of the data produced by
            extracting that feature from a volume.
        volume (Volume): a volume that the extractor will iterate over.

    """

    def __init__(self):
        self.features = {}
        self.feature_sizes = {}
        self.volume = None

    def set_volume(self, volume):
        """Set a new volume as the base for iterating."""
        self.volume = volume

    def add_feature(self, feature_name, feature_function):
        """Add a new feature to the extractor."""

        # Add the function.
        self.features[feature_name] = feature_function

        # Initialise the size as None.  Sizes are found by calling
        # self.find_feature_sizes().
        self.feature_sizes[feature_name] = None

    def find_feature_sizes(self):
        """Fill in the feature_sizes dictionary."""

        # Test that a volume is present.
        if not self.volume:
            Exception('A volume must be set to find feature sizes.')

        # Construct lists for iterating through all points.
        ranges_to_iterate = [range(size) for size in self.volume.shape]

        # Loop through all points and attempt to extract data.  Some points
        # (e.g. edge points for patches) will be invalid.
        for point in itertools.product(*ranges_to_iterate):

            # Create a list of all features for which we don't have sizes.
            rem_features = [feature_name for feature_name, feature_size in
                            self.feature_sizes.items() if feature_size is None]

            # If this list is empty, then stop.
            if len(rem_features) == 0:
                break

            # For each of these features, try the current point.
            for feature_name in rem_features:
                try:
                    data = self.extract_point_feature(point, feature_name)
                    self.feature_sizes[feature_name] = data.shape
                except:
                    continue

    def extract_point_feature(self, point, feature_name):
        """
        Extract data for a single feature from a single point.

        Args
            point (iterable): a 3-element coordinate into a volume.
            feature_name (string): the name of the feature to extract.

        Returns
            feature_data (numpy.xxx): a numpy object of some type that is the
                data for the specified feature at the specified point.

        """

        feature_function = self.features[feature_name]

        return feature_function(self.volume, point)

    def extract_point_features(self, point):
        """
        Extract data for all features from a single point.

        Args
            point (iterable): a 3-element coordinate into a volume.

        Returns
            point_data (dict): a dictionary with keys as feature names, and
                values as data corresponding to that feature.

        """

        # For the given point, evaluate all functions and return the data.
        point_data = {}
        for feature_name in self.features.keys():
            point_data[feature_name] = \
                self.extract_point_feature(point, feature_name)

        return point_data

    def iterate(self, batch_size, point_map=None):
        """
        Iterate over the currently set volume, returning data batches.

        Args
            batch_size (int): the number of points to evaluate in a batch.
            point_map (numpy.ndarray): an array that is the same size as the
                volume.  Points corresponding to non-zero elements will be
                included in the batch data.  If not supplied, all points are
                iterated over.

        Returns
            input_batch (dict): a dictionary of input data.  The first
                dimension of each value is batch_size.
            output_batch(numpy.ndarray): an array of output data.  This is
                simply a binary value indicating how each point is classified.
            point_batch (numpy.ndarray): an array of points that were used to
                generate the data.

        Notes
            Points are iterated over in a random fashion to provide even
            coverage of the entire map in each batch.  Similarly, the number of
            positives in each batch follow a common ratio - if 3/4 of the
            points being trained on are positive, then each batch will produce
            3/4 * batch_size positive points.

        """

        # Convert the point_map to a boolean array.
        if point_map is None:
            point_map = np.ones(self.volume.shape, dtype='bool')
        else:
            point_map = point_map.astype('bool')

        # Create an array containing individual maps for each category
        # (currently only pectoral/non-pectoral).
        map_types = [np.logical_and(point_map, self.volume.seg_data == 1),
                     np.logical_and(point_map, self.volume.seg_data == 0)]

        # point_sets will hold the randomised points to be sampled from each
        # map.
        point_sets = []

        # Loop through each map type to generate the random points.
        for map_type in map_types:
            map_indices = list(np.nonzero(map_type))
            point_count = map_indices[0].size
            shuffled = np.arange(point_count)
            np.random.shuffle(shuffled)
            shuffled_indices = []
            for indices in map_indices:
                shuffled_indices.append(indices[shuffled])
            point_sets.append(tuple(zip(*shuffled_indices)))

        # Create lists of statistics for each map type.
        point_counts = [len(point_set) for point_set in point_sets]
        point_ratios = [point_count / sum(point_counts)
                        for point_count in point_counts]
        batch_sizes = [int(point_ratio * batch_size)
                       for point_ratio in point_ratios]

        # Correct the last batch_size so that all add to the supplied arg.
        batch_sizes[-1] = batch_size - sum(batch_sizes[:-1])

        # Make sure that all feature sizes have been calculated.
        self.find_feature_sizes()

        # Initialise arrays for objects to return.
        input_batch = {}
        for name in self.feature_sizes.keys():
            input_batch[name] = \
                np.zeros([batch_size] + list(self.feature_sizes[name]),
                         dtype='float32')
        output_batch = np.zeros(batch_size, dtype='int32')
        point_batch = np.zeros([batch_size, len(self.volume.shape)],
                               dtype='int32')

        # Initialise a counter for the number of points successfully extracted.
        total_extracted = 0

        # point_set_counter holds the number of points that have been tested
        # under each map type.  Its sum is therefore the total number of points
        # that have been processed (including those invalid points on edges).
        point_set_counter = [0] * len(point_sets)

        # Loop through until there are no valid points left.
        while True:

            # Treat each map type separately.
            for i in range(len(map_types)):

                # For the current batch and map type, extract points until the
                # sub batch size is fulfilled.
                current_extracted = 0
                while current_extracted < batch_sizes[i]:

                    # Try to get the next point.  If it doesn't exist, stop
                    # iterating and return.
                    try:
                        point = point_sets[i][point_set_counter[i]]
                    except IndexError:
                        return

                    # Try to get the data (may raise an exception due to a
                    # feature function being invalid at the point.
                    try:
                        point_data = self.extract_point_features(point)
                        for name, data in point_data.items():
                            input_batch[name][total_extracted % batch_size] = \
                                data
                        output_batch[total_extracted % batch_size] = \
                            self.volume.seg_data[point]
                        point_batch[total_extracted % batch_size] = point

                        # Increment relevant counters.
                        current_extracted += 1
                        total_extracted += 1

                    except:
                        pass

                    # Increment the number of points processed in this
                    # particular set, regardless of success in extracting.
                    point_set_counter[i] += 1

            yield copy.deepcopy(input_batch), \
                  copy.deepcopy(output_batch), \
                  copy.deepcopy(point_batch)

    def predict(self, net, batch_size=1000, point_map=None):
        """Return a copy of the current volume, with predicted segmentation."""

        # Preallocate the array for the predicted segmentation.
        pred_seg = np.zeros(self.volume.shape)

        # Iterate through to predict values, and record.
        for input, _, points in self.iterate(batch_size, point_map=point_map):
            predicted_data = net.predict(input)
            point_indices = tuple(zip(*points))
            pred_seg[point_indices] = predicted_data

        return Volume(self.volume.mri_data, pred_seg, self.volume.landmarks)


if __name__ == '__main__':
    volume = load_volume(list_volumes()[0])
    volume.switch_plane('axial')