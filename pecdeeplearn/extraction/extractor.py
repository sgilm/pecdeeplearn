from __future__ import division

import numpy as np
import itertools
import copy

from .features import FeatureError
from .volume import Volume


class Extractor:
    """
    A class to iterate over vols and extract data for use in a neural net.

    Attributes
        features (dict): a dictionary with keys as feature names, and values as
            functions that extract data corresponding to that feature from
            a vols.
        feature_sizes (dict): a dictionary with keys as feature names, and
            values tuples that give the dimensions of the data produced by
            extracting that feature from a vols.

    """

    def __init__(self):
        self.features = {}
        self.feature_sizes = {}

    def _create_data_arrays(self, batch_size):
        """An internal method to create arrays for a specified batch size."""

        # Create the input data dictionary.
        input_batch = {}
        for name in self.feature_sizes.keys():
            data_shape = [batch_size] + list(self.feature_sizes[name])
            input_batch[name] = np.zeros(data_shape, dtype='float32')

        # Create the output and point batch arrays.
        output_batch = np.zeros(batch_size, dtype='int32')
        point_batch = np.zeros([batch_size, 3], dtype='int32')

        return input_batch, output_batch, point_batch

    @staticmethod
    def _process_input_batch(input_batch, clean_input):
        """An internal method to copy an input batch and process its format."""

        # nolearn expects single inputs to be an array, not a dictionary, so
        # apply this formatting if required to input_batch and return a copy.
        if clean_input and len(input_batch.keys()) == 1:
            return copy.deepcopy(input_batch.values()[0])
        else:
            return copy.deepcopy(input_batch)

    def add_feature(self, feature_name, feature_function):
        """Add a new feature to the extractor."""

        # Add the function.
        self.features[feature_name] = feature_function

        # Initialise the size as None.  Sizes are found by calling
        # self.find_feature_sizes().
        self.feature_sizes[feature_name] = None

    def find_feature_sizes(self, volume, point_map=None):
        """Fill in the feature_sizes dictionary using a supplied volume."""

        # If point_map is not supplied, then iterate over all points.
        if point_map is None:
            point_map = np.full(volume.shape, True, dtype='bool')

        # Construct lists for iterating through all points.
        points = zip(*np.nonzero(point_map))

        # Loop through the points and attempt to extract data.  Some points
        # (e.g. edge points for patches) will be invalid.
        for point in points:

            # Create a list of all features for which we don't have sizes.
            rem_features = [feature_name for feature_name, feature_size in
                            self.feature_sizes.items() if
                            feature_size is None]

            # If this list is empty, then stop.
            if len(rem_features) == 0:
                return

            # For each of these features, try the current point.
            for feature_name in rem_features:
                try:
                    data = self.extract_point_feature(volume,
                                                      point,
                                                      feature_name)
                except FeatureError:
                    continue

                # Add the shape.
                self.feature_sizes[feature_name] = data.shape

    def extract_point_feature(self, volume, point, feature_name):
        """
        Extract data for a single feature from a single point in a vol.

        Args
            vol (Volume): the vol to extract from.
            point (iterable): a 3-element coordinate into a vols.
            feature_name (string): the name of the feature to extract.

        Returns
            feature_data (numpy.xxx): a numpy object of some type that is the
                data for the specified feature at the specified point.

        """

        feature_function = self.features[feature_name]

        return feature_function(volume, point)

    def extract_point_features(self, volume, point):
        """
        Extract data for all features from a single point in a vol.

        Args
            vol (Volume): the vol to extract from.
            point (iterable): a 3-element coordinate into a vols.

        Returns
            point_data (dict): a dictionary with keys as feature names, and
                values as data corresponding to that feature.

        """

        # For the given point, evaluate all functions and return the data.
        point_data = {}
        for feature_name in self.features.keys():
            point_data[feature_name] = \
                self.extract_point_feature(volume, point, feature_name)

        return point_data

    def extract_from_map(self, volume, point_map, batch_size,
                         clean_input=True):
        """
        Extracts data randomly from a specified map and vol.

        Args
            vol (Volume): the vol to extract from.
            point_map (numpy.ndarray): an array that is the same size as the
                vols.  Points corresponding to non-zero elements will be
                included in the batch data.
            batch_size (int): the number of points to evaluate in a batch.

        Returns
            input_batch (dict/numpy.ndarray): if there is more than one feature
                then this is a dictionary of input data.  Otherwise, it's an
                ndarray (as required by nolearn).
            output_batch(numpy.ndarray): an array of output data.  This is
                simply a binary array indicating how each point is classified.
            point_batch (numpy.ndarray): an array of points that were used to
                generate the data.

        Notes
            Points are iterated over in a random fashion to provide even
            coverage of the entire map in each batch.

        """

        # Get the indices of the points to extract.
        map_indices = np.nonzero(point_map)

        # Get the number of points to extract.
        point_count = map_indices[0].size

        # Create a permuted array for shuffling the indices and randomising the
        # order in which the points are processed.
        permutation = np.arange(point_count)
        np.random.shuffle(permutation)

        # Shuffle the indices.
        shuffled_indices = []
        for indices in map_indices:
            shuffled_indices.append(indices[permutation])

        # Create a set of points for indexing.
        point_set = tuple(zip(*shuffled_indices))

        # Make sure all feature sizes have been calculated.
        self.find_feature_sizes(volume, point_map=point_map)

        # Initialise the arrays to return data in.
        input_batch, output_batch, point_batch = \
            self._create_data_arrays(batch_size)

        # Initialise a counter for the number of points successfully extracted.
        count = 0

        # Loop through until there are no valid points left.
        for point in point_set:

            # Try to get the data (may raise a FeatureError due to a feature
            # function being invalid at the point.
            try:
                point_data = self.extract_point_features(volume, point)

            # Skip to the next point if the feature was invalid.
            except FeatureError:
                continue

            # Copy the data into the return arrays.
            for name, data in point_data.items():
                input_batch[name][count % batch_size] = data
            output_batch[count % batch_size] = volume.seg_data[point]
            point_batch[count % batch_size] = point

            # Increment the counter.
            count += 1

            if count % batch_size == 0:
                yield self._process_input_batch(input_batch, clean_input), \
                      copy.deepcopy(output_batch), \
                      copy.deepcopy(point_batch)

        # Yield the final set of points before stopping.  This means that all
        # valid points are extracted from, but some points (in the worst case
        # batch_size - 1) will be repeated.  This isn't really a problem for
        # either prediction or training.
        yield self._process_input_batch(input_batch, clean_input), \
              copy.deepcopy(output_batch), \
              copy.deepcopy(point_batch)

    def iterate_single(self, volume, point_map, batch_size, clean_input=True):
        """
        Extract batches of data from a vol randomly, with even coverage.

        Args
            vol (Volume): the vol to extract from.
            point_map (numpy.ndarray): an array that is the same size as the
                vols.  Points corresponding to non-zero elements will be
                included in the batch data.
            batch_size (int): the number of points to evaluate in a batch.

        Returns
            input_batch (dict): a dictionary of input data.  The first
                dimension of each value is batch_size.
            output_batch(numpy.ndarray): an array of output data.  This is
                simply a binary array indicating how each point is classified.
            point_batch (numpy.ndarray): an array of points that were used to
                generate the data.

        Notes
            The number of positives in each batch follow a common ratio - if
            3/4 of the points being trained on are positive, then each batch
            will produce 3/4 * batch_size positive points.

        """

        # Create an array containing individual maps for each category
        # (currently only pectoral/non-pectoral).
        map_types = [np.logical_and(point_map, volume.seg_data == 1),
                     np.logical_and(point_map, volume.seg_data == 0)]

        # Create lists of statistics for each map type.
        point_counts = [np.sum(map_type) for map_type in map_types]
        point_ratios = [point_count / sum(point_counts)
                        for point_count in point_counts]
        sub_batch_sizes = [int(point_ratio * batch_size)
                           for point_ratio in point_ratios]

        # Correct the last sub_batch_size so that all add to the argument.
        sub_batch_sizes[-1] = batch_size - sum(sub_batch_sizes[:-1])

        # Make sure that the dictionary of feature sizes has been initialised.
        self.find_feature_sizes(volume, point_map=point_map)

        # Initialise the arrays to return data in.
        input_batch, output_batch, point_batch = \
            self._create_data_arrays(batch_size)

        # Create the individual generators for extracting data using each map.
        gens = [self.extract_from_map(volume, map_types[i], sub_batch_sizes[i],
                                      clean_input=False)
                for i in range(len(map_types))]

        # Create an array to use for insertion of individual sub_batches into
        # the returned batches.
        ins_indices = np.cumsum([0] + sub_batch_sizes)

        # Use the generators until they are exhausted.
        keep_generating = True
        while keep_generating:
            for i in range(len(gens)):

                # Try to get the next data from the current generator.
                try:
                    sub_input_batch, sub_output_batch, sub_point_batch = \
                        next(gens[i])
                except StopIteration:
                    keep_generating = False
                    break

                # Insert the data into the return array.
                ins_slice = slice(ins_indices[i], ins_indices[i + 1])
                for name, data in sub_input_batch.items():
                    input_batch[name][ins_slice] = data
                output_batch[ins_slice] = sub_output_batch
                point_batch[ins_slice] = sub_point_batch

            # If the end of a generator has not yet been reached, return the
            # data.
            if keep_generating:
                yield self._process_input_batch(input_batch, clean_input), \
                      copy.deepcopy(output_batch), \
                      copy.deepcopy(point_batch)

    def iterate_multiple(self, volumes, point_maps, batch_size,
                         clean_input=True):
        """
        Extract data from a list of vols in a balanced and random way.

        Args
            vols (list): the list of vols to extract from.
            point_maps (list): a list of maps to use for extraction.
            batch_size (int): the size of the return batches.

        Returns
            input_batch (dict): a dictionary of input data.  The first
                dimension of each value is batch_size.
            output_batch(numpy.ndarray): an array of output data.  This is
                simply a binary array indicating how each point is classified.

        """

        # Check the vols and maps data is valid.
        if len(volumes) != len(point_maps):
            raise Exception('Each vol must have a corresponding point map.')

        # Get the sub_batch_sizes for each vol based on how many points are
        # being extracted from it.
        point_counts = [np.sum(point_map) for point_map in point_maps]
        point_ratios = [point_count / sum(point_counts)
                        for point_count in point_counts]
        sub_batch_sizes = [int(point_ratio * batch_size)
                           for point_ratio in point_ratios]

        # Make sure that the dictionary of feature sizes has been initialised.
        for volume, point_map in zip(volumes, point_maps):
            self.find_feature_sizes(volume, point_map=point_map)

        # Initialise the arrays to return data in.
        input_batch, output_batch, point_batch = \
            self._create_data_arrays(batch_size)

        # Create the generators for each vol.
        gens = [self.iterate_single(volumes[i], point_maps[i],
                                    sub_batch_sizes[i], clean_input=False)
                for i in range(len(volumes))]

        # Create an array to use for insertion of individual sub_batches into
        # the returned batches.
        ins_indices = np.cumsum([0] + sub_batch_sizes)

        # Use the generators until they are exhausted.
        keep_generating = True
        while keep_generating:
            for i in range(len(gens)):

                # Try to get the next sub batch.
                try:
                    sub_input_batch, sub_output_batch, _ = next(gens[i])
                except StopIteration:
                    keep_generating = False
                    break

                # Insert the data into the return array.
                ins_slice = slice(ins_indices[i], ins_indices[i + 1])
                for name, data in sub_input_batch.items():
                    input_batch[name][ins_slice] = data
                output_batch[ins_slice] = sub_output_batch

            # If the end of a generator has not yet been reached, return the
            # data.
            if keep_generating:

                yield self._process_input_batch(input_batch, clean_input),\
                      copy.deepcopy(output_batch)

    def predict(self, net, volume, batch_size):
        """Return a copy of the supplied vol with predicted segmentation."""

        # Preallocate the array for the predicted segmentation.
        predicted_seg = np.zeros(volume.shape)

        # Iterate through to predict values, and record.
        for input_batch, _, point_batch in \
                self.extract_from_map(volume, np.ones(volume.shape),
                                      batch_size):

            # Predict and assign the values to the new volume.
            predicted_data = net.predict(input_batch)
            point_indices = tuple(zip(*point_batch))
            predicted_seg[point_indices] = predicted_data.reshape(batch_size)

        return Volume(volume.mri_data, predicted_seg,
                      copy.deepcopy(volume.landmarks), volume.orientation)
