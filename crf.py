from __future__ import division

import pystruct.models as models
import pystruct.learners as learners
import numpy as np
import pecdeeplearn as pdl
import data_path
import itertools
import copy


if __name__ == '__main__':

    # Setup experiment access.
    exp = pdl.utils.Experiment(data_path.get())
    exp.load_experiment('crf')

    # Record parameters.
    exp.add_param('max_points_per_volume', 100)
    exp.add_param('margins', (12, 12, 12))
    exp.add_param('model_shape', (25, 25, 25))
    exp.add_param('batch_size', 10)

    # Load the predicted and actual volumes.
    train_vol_names = ['VL00033', 'VL00034']
    test_vol_names = ['VL00069']
    other_vol_names = [vol_name for vol_name in exp.list_volumes()
                       if vol_name not in train_vol_names
                       and vol_name not in test_vol_names]

    train_vols_actual = [exp.load_volume(vol, experiment=False)[:, :, ]
                         for vol in train_vol_names]
    train_vols_predicted = [exp.load_volume(vol, experiment=True)
                            for vol in train_vol_names]
    test_vols_actual = [exp.load_volume(vol, experiment=False)
                        for vol in test_vol_names]
    test_vols_predicted = [exp.load_volume(vol, experiment=True)
                           for vol in test_vol_names]
    other_vols = [exp.load_volume(vol) for vol in other_vol_names]
    # pdl.utils.standardise_volumes(
    #     train_vols_actual + test_vols_actual + other_vols)

    # Construct training maps.
    training_maps = [
        pdl.extraction.half_half_map(
            vol,
            max_points=exp.params['max_points_per_volume'],
            margins=exp.params['margins']
        )
        for vol in train_vols_actual]

    # Create extractor to retrieve patch values.
    ext = pdl.extraction.Extractor()
    ext.add_feature(
        feature_name='patch',
        feature_function=lambda volume, point:
        pdl.extraction.patch(volume, point, exp.params['model_shape'])
    )

    # Define a numpy array with ones in the positions corresponding to nodes
    # that will be used in the model.
    graph_points = np.zeros(exp.params['model_shape'], dtype='bool')
    for axis, size in list(enumerate(exp.params['model_shape'])):
        indices = list(np.array(exp.params['model_shape']) // 2)
        indices[axis] = slice(0, None)
        graph_points[tuple(indices)] = True

    # Build edges.
    edges = []
    base_points = np.array(np.nonzero(graph_points))
    indices_and_points = list(enumerate(base_points.T))
    for (from_index, from_point), (to_index, to_point) in \
            itertools.product(indices_and_points, indices_and_points):
        dist = np.linalg.norm(to_point - from_point)
        if dist == 1 and \
                graph_points[tuple(from_point)] and \
                graph_points[tuple(to_point)]:
            edges.append([from_index, to_index])
    edges = np.array(edges)
    centre_index = np.unique(edges, return_counts=True)[1].argmax()

    # Build training dataset.
    training_input = []
    training_output = []
    for train_vol_actual, train_vol_predicted, training_map in \
            zip(train_vols_actual, train_vols_predicted, training_maps):
        for _, _, point_batch in ext.iterate_single(
                train_vol_actual, training_map, exp.params['batch_size']):
            for point in point_batch:
                points = point + (base_points.T -
                                  np.array(exp.params['model_shape']) // 2)
                indices = tuple(points.T)
                intensity = train_vol_actual.mri_data[indices].flatten()
                predicted_seg = train_vol_predicted.seg_data[indices].flatten()

                training_input.append(
                    (np.stack((predicted_seg, intensity), axis=1), edges)
                )

                actual_seg = \
                    train_vol_actual.seg_data[indices].astype('int8').flatten()
                training_output.append(actual_seg)

    # Construct model and learner.
    model = models.GraphCRF(inference_method='qpbo')
    learner = learners.NSlackSSVM(
        model,
        verbose=2,
        max_iter=100,
        n_jobs=-1,
        tol=0.01,
        show_loss_every=5,
        inactive_threshold=1e-3,
        inactive_window=10,
        batch_size=100
    )

    # Learn model params.
    learner.fit(training_input, training_output)

    # Predict segmentation.
    for test_vol_predicted in test_vols_predicted:
        testing_input = []
        slices_to_predict = \
            [slice(margin, size - margin)
             for margin, size
             in zip(exp.params['margins'], test_vol_predicted.shape)]
        margined_prob_seg = test_vol_predicted.seg_data[slices_to_predict]
        margined_seg = np.around(margined_prob_seg).astype('bool')

        points_to_test = np.array(np.nonzero(margined_seg)).T + \
                         np.array(exp.params['margins'])

        for point in points_to_test:
            points = point + (base_points.T -
                              np.array(exp.params['model_shape']) // 2)
            indices = tuple(points.T)
            intensity = test_vol_predicted.mri_data[indices].flatten()
            predicted_seg = test_vol_predicted.seg_data[indices].flatten()

            testing_input.append(
                (np.stack((predicted_seg, intensity), axis=1), edges)
            )

        # Predict.
        batch_size = 10000
        crf_seg = np.zeros(test_vol_predicted.shape)
        for batch_start in \
                range(0, len(testing_input) - batch_size, batch_size):
            testing_input_batch = \
                testing_input[batch_start:batch_start + batch_size]
            points_to_test_batch = \
                points_to_test[batch_start:batch_start + batch_size]

            crf_outputs = np.array(learner.predict(testing_input_batch))
            crf_seg[tuple(points_to_test_batch.T)] = crf_outputs[:, centre_index]

        crf_vol = copy.deepcopy(test_vol_predicted)
        crf_vol.name += '_crf'
        crf_vol.seg_data = crf_seg

        exp.export_nii(crf_vol)