from __future__ import division

import pecdeeplearn as pdl
import data_path
import time
import numpy as np


# Create an experiment object to keep track of parameters and facilitate data
# loading and saving.
exp = pdl.utils.Experiment(data_path.get())
exp.load_experiment('structured')
exp.add_param('num_training_volumes', 45)
exp.add_param('max_points_per_volume', 25000)
exp.add_param('margins', (37, 37, 37))
exp.add_param('local_a_patch_shape', [25, 25, 1])
exp.add_param('local_c_patch_shape', [25, 1, 25])
exp.add_param('local_s_patch_shape', [1, 25, 25])
exp.add_param('prob_a_source_patch_shape', [75, 75, 1])
exp.add_param('prob_c_source_patch_shape', [75, 1, 75])
exp.add_param('prob_s_source_patch_shape', [1, 75, 75])
exp.add_param('prob_a_target_patch_shape', [25, 25, 1])
exp.add_param('prob_c_target_patch_shape', [25, 1, 25])
exp.add_param('prob_s_target_patch_shape', [1, 25, 25])
exp.add_param('local_patch_input_shape', [1, 25, 25])
exp.add_param('prob_patch_input_shape', [1, 25, 25])
exp.add_param('landmark_1', 'Sternal angle')
exp.add_param('landmark_2', 'Left nipple')
exp.add_param('landmark_3', 'Right nipple')
exp.add_param('local_a_patch_conv1_filter_size', (3, 3))
exp.add_param('local_a_patch_conv1_num_filters', 64)
exp.add_param('local_a_patch_pool1_pool_size', (2, 2))
exp.add_param('local_a_patch_conv2_filter_size', (3, 3))
exp.add_param('local_a_patch_conv2_num_filters', 128)
exp.add_param('local_a_patch_pool2_pool_size', (2, 2))
exp.add_param('local_a_patch_conv3_filter_size', (3, 3))
exp.add_param('local_a_patch_conv3_num_filters', 256)
exp.add_param('local_a_patch_pool3_pool_size', (2, 2))
exp.add_param('local_c_patch_conv1_filter_size', (3, 3))
exp.add_param('local_c_patch_conv1_num_filters', 64)
exp.add_param('local_c_patch_pool1_pool_size', (2, 2))
exp.add_param('local_c_patch_conv2_filter_size', (3, 3))
exp.add_param('local_c_patch_conv2_num_filters', 128)
exp.add_param('local_c_patch_pool2_pool_size', (2, 2))
exp.add_param('local_c_patch_conv3_filter_size', (3, 3))
exp.add_param('local_c_patch_conv3_num_filters', 256)
exp.add_param('local_c_patch_pool3_pool_size', (2, 2))
exp.add_param('local_s_patch_conv1_filter_size', (3, 3))
exp.add_param('local_s_patch_conv1_num_filters', 64)
exp.add_param('local_s_patch_pool1_pool_size', (2, 2))
exp.add_param('local_s_patch_conv2_filter_size', (3, 3))
exp.add_param('local_s_patch_conv2_num_filters', 128)
exp.add_param('local_s_patch_pool2_pool_size', (2, 2))
exp.add_param('local_s_patch_conv3_filter_size', (3, 3))
exp.add_param('local_s_patch_conv3_num_filters', 256)
exp.add_param('local_s_patch_pool3_pool_size', (2, 2))
exp.add_param('prob_a_patch_conv1_filter_size', (3, 3))
exp.add_param('prob_a_patch_conv1_num_filters', 64)
exp.add_param('prob_a_patch_pool1_pool_size', (2, 2))
exp.add_param('prob_a_patch_conv2_filter_size', (3, 3))
exp.add_param('prob_a_patch_conv2_num_filters', 128)
exp.add_param('prob_a_patch_pool2_pool_size', (2, 2))
exp.add_param('prob_a_patch_conv3_filter_size', (3, 3))
exp.add_param('prob_a_patch_conv3_num_filters', 256)
exp.add_param('prob_a_patch_pool3_pool_size', (2, 2))
exp.add_param('prob_c_patch_conv1_filter_size', (3, 3))
exp.add_param('prob_c_patch_conv1_num_filters', 64)
exp.add_param('prob_c_patch_pool1_pool_size', (2, 2))
exp.add_param('prob_c_patch_conv2_filter_size', (3, 3))
exp.add_param('prob_c_patch_conv2_num_filters', 128)
exp.add_param('prob_c_patch_pool2_pool_size', (2, 2))
exp.add_param('prob_c_patch_conv3_filter_size', (3, 3))
exp.add_param('prob_c_patch_conv3_num_filters', 256)
exp.add_param('prob_c_patch_pool3_pool_size', (2, 2))
exp.add_param('prob_s_patch_conv1_filter_size', (3, 3))
exp.add_param('prob_s_patch_conv1_num_filters', 64)
exp.add_param('prob_s_patch_pool1_pool_size', (2, 2))
exp.add_param('prob_s_patch_conv2_filter_size', (3, 3))
exp.add_param('prob_s_patch_conv2_num_filters', 128)
exp.add_param('prob_s_patch_pool2_pool_size', (2, 2))
exp.add_param('prob_s_patch_conv3_filter_size', (3, 3))
exp.add_param('prob_s_patch_conv3_num_filters', 256)
exp.add_param('prob_s_patch_pool3_pool_size', (2, 2))
exp.add_param('landmark_1_dense1_num_units', 100)
exp.add_param('landmark_1_dense2_num_units', 100)
exp.add_param('landmark_1_dense3_num_units', 100)
exp.add_param('landmark_2_dense1_num_units', 100)
exp.add_param('landmark_2_dense2_num_units', 100)
exp.add_param('landmark_2_dense3_num_units', 100)
exp.add_param('landmark_3_dense1_num_units', 100)
exp.add_param('landmark_3_dense2_num_units', 100)
exp.add_param('landmark_3_dense3_num_units', 100)
exp.add_param('join_dense1_num_units', 1000)
exp.add_param('batch_size', 5000)
exp.add_param('update_learning_rate', 0.001)
exp.add_param('update_momentum', 0.9)
exp.add_param('max_epochs', 100)
exp.add_param('validation_prop', 0.2)
exp.add_param('prediction_margins', (15, 15, 15))

# List and load all volumes.
vol_list = exp.list_volumes()
test_vol_names = ['VL00027', 'VL00032', 'VL00033', 'VL00035', 'VL00042',
                  'VL00047', 'VL00049', 'VL00056', 'VL00066', 'VL00067',
                  'VL00070', 'VL00074', 'VL00080', 'VL00090', 'VL00096']
for vol_name in test_vol_names:
    try:
        vol_list.remove(vol_name)
        vol_list.append(vol_name)
    except ValueError:
        pass
vols = [(exp.load_volume(vol, experiment=False),
         exp.load_volume(vol, experiment=True, suffix='_prob'))
        for vol in vol_list]

# Load the experiment for saving the predictions to.
exp.load_experiment(
    'secondary_triple_layer_acs_conv_three_landmark_targeted_1'
)

# Standardise the data.
pdl.utils.standardise_volumes([actual for actual, predicted in vols])

# Use a dirty hack to help with extracting probability data later down the
# line.
for actual, predicted in vols:
    actual.prob_seg_data = predicted.seg_data

# Split into a training set and testing set.
training_vols = vols[:exp.params['num_training_volumes']]
testing_vols = vols[exp.params['num_training_volumes']:]

# Create an Extractor.
ext = pdl.extraction.Extractor()

# Add features.
ext.add_feature(
    feature_name='local_a_patch',
    feature_function=lambda volume, point:
    pdl.extraction.patch(volume, point, exp.params['local_a_patch_shape'])
)
ext.add_feature(
    feature_name='local_c_patch',
    feature_function=lambda volume, point:
    pdl.extraction.patch(volume, point, exp.params['local_c_patch_shape'])
)
ext.add_feature(
    feature_name='local_s_patch',
    feature_function=lambda volume, point:
    pdl.extraction.patch(volume, point, exp.params['local_s_patch_shape'])
)
ext.add_feature(
    feature_name='prob_a_patch',
    feature_function=lambda volume, point:
    pdl.extraction.scaled_patch(volume, point,
                                exp.params['prob_a_source_patch_shape'],
                                exp.params['prob_a_target_patch_shape'],
                                prob_seg=True)
)
ext.add_feature(
    feature_name='prob_c_patch',
    feature_function=lambda volume, point:
    pdl.extraction.scaled_patch(volume, point,
                                exp.params['prob_c_source_patch_shape'],
                                exp.params['prob_c_target_patch_shape'],
                                prob_seg=True)
)
ext.add_feature(
    feature_name='prob_s_patch',
    feature_function=lambda volume, point:
    pdl.extraction.scaled_patch(volume, point,
                                exp.params['prob_s_source_patch_shape'],
                                exp.params['prob_s_target_patch_shape'],
                                prob_seg=True)
)
ext.add_feature(
    feature_name='landmark_1',
    feature_function=lambda volume, point:
    pdl.extraction.landmark_displacement(
        volume, point, exp.params['landmark_1'])
)
ext.add_feature(
    feature_name='landmark_2',
    feature_function=lambda volume, point:
    pdl.extraction.landmark_displacement(
        volume, point, exp.params['landmark_2'])
)
ext.add_feature(
    feature_name='landmark_3',
    feature_function=lambda volume, point:
    pdl.extraction.landmark_displacement(
        volume, point, exp.params['landmark_3'])
)

# Create the net.
net = exp.unpickle_network('net')

# Perform predictions on all testing volumes in the set.
actual_testing_vols = [actual for actual, predicted in testing_vols]
print('Beginning predictions.\n')
prediction_start_time = time.time()
for i, testing_vol in list(enumerate(actual_testing_vols)):

    # Create bounds to avoid unnecessary prediction.
    bounds = list(testing_vol.bounding_box(
        margins=exp.params['prediction_margins']))
    for j, (size, margin) in enumerate(zip(testing_vol.shape,
                                           exp.params['margins'])):
        bounds[0][j] = max(bounds[0][j], margin)
        bounds[1][j] = min(bounds[1][j], size - margin - 1)

    # Perform the prediction on the current testing volume.
    print("Predicting on volume " + testing_vol.name + ".")
    predicted_vol = ext.predict(
        net,
        testing_vol,
        exp.params['batch_size'],
        bounds=bounds
    )

    # Save the prediction probabilities for comparison.
    predicted_name = predicted_vol.name
    predicted_vol.name += "_prob"
    exp.export_nii(predicted_vol)

    # Save the rounded segmentation.
    predicted_vol.name = predicted_name
    predicted_vol.seg_data = np.around(predicted_vol.seg_data).astype('int16')
    exp.export_nii(predicted_vol)

    # Print prediction progress.
    pdl.utils.print_progress(time.time() - prediction_start_time,
                             i + 1,
                             len(testing_vols))

# Record the parameters and results.
exp.record()
