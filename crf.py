from __future__ import division

import pystruct.models as models
import pystruct.learners as learners
import numpy as np
import pecdeeplearn as pdl
import data_path
import itertools
import copy
import time


def extract_crf_data(vols, point_maps, offsets, edges, gen_batch_size=100):

    # Create extractor that will be used for retrieving point indices only
    # (not intensities).
    ext = pdl.extraction.Extractor()
    ext.add_feature(
        feature_name='dummy',
        feature_function=lambda volume, point:
        pdl.extraction.point_offset(volume, point, (0, 0, 0))
    )

    # Create generators for iterating through all volumes.  The batch size is
    # arbitrary.
    gens = \
        [ext.iterate_single(actual_vol, point_map, gen_batch_size)
         for (actual_vol, predicted_vol), point_map in zip(vols, point_maps)]

    # Define the names of the landmarks whose displacements are to be node
    # properties.
    landmark_names = ['Sternal angle', 'Left nipple', 'Right nipple']

    # Iterate through all the supplied volumes.
    input_data = []
    output_data = []
    point_data = []
    for gen, (actual_vol, predicted_vol) in zip(gens, vols):
        for _, _, point_batch in gen:
            for point in np.array(point_batch):

                # Form the points and indices which need to be extracted for
                # the model around the current point.
                points_to_extract = point + offsets
                indices_to_extract = tuple(points_to_extract.T)

                # Get the intensity, predicted segmentation and landmark data
                # as node properties.
                intensity = actual_vol.mri_data[indices_to_extract].flatten()
                predicted_seg = \
                    predicted_vol.seg_data[indices_to_extract].flatten()
                landmark_data = []
                for landmark_name in landmark_names:
                    new_landmark_data = pdl.extraction.landmark_displacement(
                        actual_vol,
                        points_to_extract,
                        landmark_name
                    )
                    landmark_data.append(new_landmark_data)
                landmark_data = np.concatenate(landmark_data, axis=1)

                # Append the node properties to the input data list.
                input_data.append(
                    (np.concatenate(
                        (np.stack((predicted_seg, intensity), axis=1),
                         landmark_data),
                        axis=1), edges)
                )

                # Retrieve the actual segmentation values and append them to
                # the output data list.
                actual_seg = actual_vol.seg_data[indices_to_extract]. \
                    astype('int8').flatten()
                output_data.append(actual_seg)

                # Append the point data.
                point_data.append(point)

    return input_data, output_data, point_data


def build_points_and_edges(graph_points):

    # Get the indices (within the model itself) of the points to be included as
    # nodes.
    base_points = np.array(np.nonzero(graph_points))
    indices_and_points = list(enumerate(base_points.T))

    # Build edges by looping through each pair of points.
    edges = []
    for (from_index, from_point), (to_index, to_point) in \
            itertools.product(indices_and_points, repeat=2):
        dist = np.linalg.norm(to_point - from_point)

        # If two points are adjacent, add an edge.
        if dist == 1:
            edges.append([from_index, to_index])

    edges = np.array(edges)

    return base_points, edges


def strand_model(model_shape):

    # Define a numpy array with ones in the positions corresponding to nodes
    # that will be used in the model.
    graph_points = np.zeros(model_shape, dtype='bool')
    for axis, size in enumerate(model_shape):
        indices = list(np.array(model_shape) // 2)
        indices[axis] = slice(0, None)
        graph_points[tuple(indices)] = True

    base_points, edges = build_points_and_edges(graph_points)

    # Get the array of offsets that can be added to a single point to give the
    # indicies of all surrounding points that are in the model.
    offsets = (base_points.T - np.array(model_shape) // 2)

    # Find the index of the point at the centre of the model.
    centre_index = np.where(~offsets.any(axis=1))[0][0]

    return offsets, edges, centre_index


def cube_model(model_shape):

    # Define a numpy array with ones in the positions corresponding to nodes
    # that will be used in the model.
    graph_points = np.ones(model_shape, dtype='bool')

    base_points, edges = build_points_and_edges(graph_points)

    # Get the array of offsets that can be added to a single point to give the
    # indicies of all surrounding points that are in the model.
    offsets = (base_points.T - np.array(model_shape) // 2)

    # Find the index of the point at the centre of the model.
    centre_index = np.where(~offsets.any(axis=1))[0][0]

    return offsets, edges, centre_index


if __name__ == '__main__':

    # Setup experiment access.
    exp = pdl.utils.Experiment(data_path.get())
    exp.add_param('max_points_per_volume', 1000)
    exp.add_param('margins', (12, 12, 12))
    exp.add_param('model_shape', (25, 25, 25))
    exp.add_param('prediction_batch_size', 1000)

    # Load the predicted and actual volumes.
    train_vol_names = ['VL00028', 'VL00077']
    test_vol_names = ['VL00093']
    other_vol_names = [vol_name for vol_name in exp.list_volumes()
                       if vol_name not in train_vol_names
                       and vol_name not in test_vol_names]

    # Load volumes from the experiment results being tested on.
    exp.load_experiment('best')
    train_vols = [(exp.load_volume(vol, experiment=False),
                   exp.load_volume(vol, experiment=True, suffix='_prob'))
                  for vol in train_vol_names]
    test_vols = [(exp.load_volume(vol, experiment=False),
                  exp.load_volume(vol, experiment=True, suffix='_prob'))
                 for vol in test_vol_names]
    other_vols = [exp.load_volume(vol) for vol in other_vol_names]
    pdl.utils.standardise_volumes(
        [actual for actual, predicted in train_vols + test_vols] + other_vols
    )

    # Create a new experiment for outputting results of the crf segmentation.
    exp.create_experiment('crf')

    # Construct training maps.
    training_maps = \
        [pdl.extraction.actual_predicted_map(
            actual,
            predicted,
            exp.params['max_points_per_volume'],
            margins=exp.params['margins']
        )
        for actual, predicted in train_vols]

    # Get the required data from the graphical model being used.
    offsets, edges, centre_index = strand_model(exp.params['model_shape'])

    # Build training dataset.
    training_input, training_output, _ = extract_crf_data(
        train_vols,
        training_maps,
        offsets,
        edges
    )

    # Construct model and learner.
    model = models.GraphCRF(inference_method='qpbo')
    learner = learners.NSlackSSVM(
        model,
        verbose=2,
        max_iter=10000,
        n_jobs=-1,
        tol=0.001,
        show_loss_every=5,
        inactive_threshold=1e-3,
        inactive_window=10,
        batch_size=100
    )

    # Learn model parameters.
    start_time = time.time()
    learner.fit(training_input, training_output)
    exp.add_result('learning_duration', time.time() - start_time)

    # Predict a segmentation using model inference.
    for test_vol_actual, test_vol_predicted in test_vols:

        # Only predict on the bounding box of the predicted volume.
        bounds = test_vol_predicted.bounding_box()
        slices = []
        for margin, size, (start, stop) in zip(exp.params['margins'],
                                               test_vol_predicted.shape,
                                               np.array(bounds).T):
            start = max(start, margin)
            stop = min(stop + 1, size - margin)
            slices.append(slice(start ,stop))

        # Form the testing map and extract the data to predict on.
        testing_map = np.zeros(test_vol_predicted.shape, dtype='bool')
        testing_map[slices] = True
        testing_input, _, testing_points = extract_crf_data(
            [(test_vol_actual, test_vol_predicted)],
            [testing_map],
            offsets,
            edges
        )

        # Predict on the testing volume.
        batch_size = exp.params['prediction_batch_size']
        crf_seg_count = np.zeros(test_vol_predicted.shape)
        crf_prediction_count = np.zeros(test_vol_predicted.shape)
        for batch_start in range(0, len(testing_input), batch_size):

            # Extract a batch of data and perform inference.
            batch_end = batch_start + batch_size
            testing_input_batch = \
                testing_input[batch_start:min(batch_end, len(testing_input))]
            testing_points_batch = np.array(
                testing_points[batch_start:min(batch_end, len(testing_input))])

            # Accumulate the point classifications and total inferences on each
            # point.
            crf_outputs = np.array(learner.predict(testing_input_batch))
            crf_seg_count[tuple(testing_points_batch.T)] += \
                crf_outputs[:, centre_index]
            crf_prediction_count[tuple(testing_points_batch.T)] += 1

        # Calculate the proportion of tests that each voxel is classified as a
        # pectoral muscle.
        crf_props = crf_seg_count / crf_prediction_count

        # Form the predicted volume.
        crf_vol = copy.deepcopy(test_vol_actual)
        crf_vol.name += '_crf'
        crf_vol.seg_data = np.around(crf_props).astype('bool')

        # Export the segmentation.
        exp.export_nii(crf_vol)

    exp.record()