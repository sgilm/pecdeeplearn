from __future__ import division

import pystruct.models as models
import pystruct.learners as learners
import numpy as np
import pecdeeplearn as pdl
import data_path
import itertools


if __name__ == '__main__':

    # Setup experiment access.
    exp = pdl.utils.Experiment(data_path.get())
    exp.load_experiment(
        'single_local_a_conv_single_context_a_conv_triple_landmark_2')
    predicted = exp.load_volume('VL00069', experiment=True)
    actual = exp.load_volume('VL00069', experiment=False)

    # Extract 1000 points.
    point_map = pdl.extraction.half_half_map(predicted, max_points=5000)

    # Define the neighbourhood kernel and reach from centre to outside.
    neighbourhood = [3, 3, 3]
    reach = 3 // 2

    # Create extractor to retrieve patch values.
    ext = pdl.extraction.Extractor()
    ext.add_feature(
        feature_name='flat_patch',
        feature_function=lambda volume, point:
        pdl.extraction.flat_patch(volume, point, neighbourhood)
    )

    # Build edges.
    edges = []
    points = list(itertools.product(*[range(size) for size in neighbourhood]))
    for from_index, from_point in enumerate(points):
        for to_index, to_point in enumerate(points):
            dist = np.linalg.norm(np.array(to_point) - np.array(from_point))
            if dist == 1:
                edges.append([from_index, to_index])
    edges = np.array(edges)

    # Build training dataset.
    input_train = []
    output_train = []
    for input_batch, _, point_batch in \
            ext.iterate_single(predicted, point_map, 1000):
        for point, input_instance in zip(point_batch, input_batch):
            slices = tuple(
                [slice(index - reach, index + reach + 1) for index in point]
            )
            pred_seg = predicted.seg_data[slices].reshape(27, 1)
            intensity = input_instance.reshape(27, 1)
            input_train.append((np.concatenate((pred_seg, intensity), axis=1), edges))

            act_seg = actual.seg_data[slices].astype('int64').flatten()
            output_train.append(act_seg)

    # Construct model and learner.
    model = models.GraphCRF(inference_method='qpbo')
    ssvm = learners.NSlackSSVM(
        model, verbose=2, max_iter=10, n_jobs=-1,
        tol=0.0001, show_loss_every=5,
        inactive_threshold=1e-3, inactive_window=10, batch_size=100)

    # Learn model params.
    ssvm.fit(input_train, output_train)


    input_test = []
    output_test = []
    pred_points = list(
        itertools.product(np.arange(100, 300), np.arange(100, 300)))
    pred_points = [list(point) for point in pred_points]
    for point in pred_points:
        point.append(100)
    for point in pred_points:
        slices = tuple(
            [slice(index - reach, index + reach + 1) for index in point]
        )
        pred_seg = predicted.seg_data[slices].reshape(27, 1)
        intensity = predicted.mri_data[slices].reshape(27, 1)
        input_test.append(
            (np.concatenate((pred_seg, intensity), axis=1), edges))

        act_seg = actual.seg_data[slices].astype('int64').flatten()
        output_test.append(act_seg)


    # Predict.
    output_pred = ssvm.predict(input_test)

    result = np.zeros((448, 448))
    for i, point in enumerate(pred_points):
        result[tuple(point[:2])] = output_pred[i][13]

    import scipy.misc

    scipy.misc.toimage(result).show()

    x = 1