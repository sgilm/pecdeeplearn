from __future__ import division

import lasagne
import numpy as np
import nolearn.lasagne
import pecdeeplearn as pdl
import data_path
import time


# Create an experiment object to keep track of parameters and facilitate data
# loading and saving.
exp = pdl.utils.Experiment(data_path.get())
exp.create_experiment('triple_layer_acs_conv_three_landmark_targeted_context')
exp.add_param('num_training_volumes', 45)
exp.add_param('max_points_per_volume', 25000)
exp.add_param('margins', (25, 25, 25))
exp.add_param('local_a_patch_shape', [25, 25, 1])
exp.add_param('local_c_patch_shape', [25, 1, 25])
exp.add_param('local_s_patch_shape', [1, 25, 25])
exp.add_param('context_a_patch_shape', [51, 51, 1])
exp.add_param('context_c_patch_shape', [51, 1, 51])
exp.add_param('context_s_patch_shape', [1, 51, 51])
exp.add_param('local_patch_input_shape', [1, 25, 25])
exp.add_param('context_patch_input_shape', [1, 25, 25])
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
exp.add_param('context_a_patch_conv1_filter_size', (3, 3))
exp.add_param('context_a_patch_conv1_num_filters', 64)
exp.add_param('context_a_patch_pool1_pool_size', (2, 2))
exp.add_param('context_a_patch_conv2_filter_size', (3, 3))
exp.add_param('context_a_patch_conv2_num_filters', 128)
exp.add_param('context_a_patch_pool2_pool_size', (2, 2))
exp.add_param('context_a_patch_conv3_filter_size', (3, 3))
exp.add_param('context_a_patch_conv3_num_filters', 256)
exp.add_param('context_a_patch_pool3_pool_size', (2, 2))
exp.add_param('context_c_patch_conv1_filter_size', (3, 3))
exp.add_param('context_c_patch_conv1_num_filters', 64)
exp.add_param('context_c_patch_pool1_pool_size', (2, 2))
exp.add_param('context_c_patch_conv2_filter_size', (3, 3))
exp.add_param('context_c_patch_conv2_num_filters', 128)
exp.add_param('context_c_patch_pool2_pool_size', (2, 2))
exp.add_param('context_c_patch_conv3_filter_size', (3, 3))
exp.add_param('context_c_patch_conv3_num_filters', 256)
exp.add_param('context_c_patch_pool3_pool_size', (2, 2))
exp.add_param('context_s_patch_conv1_filter_size', (3, 3))
exp.add_param('context_s_patch_conv1_num_filters', 64)
exp.add_param('context_s_patch_pool1_pool_size', (2, 2))
exp.add_param('context_s_patch_conv2_filter_size', (3, 3))
exp.add_param('context_s_patch_conv2_num_filters', 128)
exp.add_param('context_s_patch_pool2_pool_size', (2, 2))
exp.add_param('context_s_patch_conv3_filter_size', (3, 3))
exp.add_param('context_s_patch_conv3_num_filters', 256)
exp.add_param('context_s_patch_pool3_pool_size', (2, 2))
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
exp.add_param('prediction_margins', (25, 25, 25))

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
vols = [exp.load_volume(vol) for vol in vol_list]

# Standardise the data.
pdl.utils.standardise_volumes(vols)

# Split into a training set and testing set.
training_vols = vols[:exp.params['num_training_volumes']]
testing_vols = vols[exp.params['num_training_volumes']:]

# Create training maps.
training_maps = [
    pdl.extraction.targeted_map(
        vol,
        max_points=exp.params['max_points_per_volume'],
        margins=exp.params['margins']
    )
    for vol in training_vols]

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
    feature_name='context_a_patch',
    feature_function=lambda volume, point:
    pdl.extraction.patch(volume, point, exp.params['context_a_patch_shape'])
)
ext.add_feature(
    feature_name='context_c_patch',
    feature_function=lambda volume, point:
    pdl.extraction.patch(volume, point, exp.params['context_c_patch_shape'])
)
ext.add_feature(
    feature_name='context_s_patch',
    feature_function=lambda volume, point:
    pdl.extraction.patch(volume, point, exp.params['context_s_patch_shape'])
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
net = nolearn.lasagne.NeuralNet(
    layers=[

        # Layers for the axial local patch.
        (lasagne.layers.InputLayer,
         {'name': 'local_a_patch',
          'shape': tuple([None] + exp.params['local_patch_input_shape'])}),
        (lasagne.layers.Conv2DLayer,
         {'name': 'local_a_patch_conv1',
          'num_filters': exp.params['local_a_patch_conv1_num_filters'],
          'filter_size': exp.params['local_a_patch_conv1_filter_size']}),
        (lasagne.layers.MaxPool2DLayer,
         {'name': 'local_a_patch_pool1',
          'pool_size': exp.params['local_a_patch_pool1_pool_size']}),
        (lasagne.layers.Conv2DLayer,
         {'name': 'local_a_patch_conv2',
          'num_filters': exp.params['local_a_patch_conv2_num_filters'],
          'filter_size': exp.params['local_a_patch_conv2_filter_size']}),
        (lasagne.layers.MaxPool2DLayer,
         {'name': 'local_a_patch_pool2',
          'pool_size': exp.params['local_a_patch_pool2_pool_size']}),
        (lasagne.layers.Conv2DLayer,
         {'name': 'local_a_patch_conv3',
          'num_filters': exp.params['local_a_patch_conv3_num_filters'],
          'filter_size': exp.params['local_a_patch_conv3_filter_size']}),
        (lasagne.layers.MaxPool2DLayer,
         {'name': 'local_a_patch_pool3',
          'pool_size': exp.params['local_a_patch_pool3_pool_size']}),
        (lasagne.layers.FlattenLayer, {'name': 'local_a_patch_flat1'}),

        # Layers for the coronal local patch.
        (lasagne.layers.InputLayer,
         {'name': 'local_c_patch',
          'shape': tuple([None] + exp.params['local_patch_input_shape'])}),
        (lasagne.layers.Conv2DLayer,
         {'name': 'local_c_patch_conv1',
          'num_filters': exp.params['local_c_patch_conv1_num_filters'],
          'filter_size': exp.params['local_c_patch_conv1_filter_size']}),
        (lasagne.layers.MaxPool2DLayer,
         {'name': 'local_c_patch_pool1',
          'pool_size': exp.params['local_c_patch_pool1_pool_size']}),
        (lasagne.layers.Conv2DLayer,
         {'name': 'local_c_patch_conv2',
          'num_filters': exp.params['local_c_patch_conv2_num_filters'],
          'filter_size': exp.params['local_c_patch_conv2_filter_size']}),
        (lasagne.layers.MaxPool2DLayer,
         {'name': 'local_c_patch_pool2',
          'pool_size': exp.params['local_c_patch_pool2_pool_size']}),
        (lasagne.layers.Conv2DLayer,
         {'name': 'local_c_patch_conv3',
          'num_filters': exp.params['local_c_patch_conv3_num_filters'],
          'filter_size': exp.params['local_c_patch_conv3_filter_size']}),
        (lasagne.layers.MaxPool2DLayer,
         {'name': 'local_c_patch_pool3',
          'pool_size': exp.params['local_c_patch_pool3_pool_size']}),
        (lasagne.layers.FlattenLayer, {'name': 'local_c_patch_flat1'}),

        # Layers for the sagittal local patch.
        (lasagne.layers.InputLayer,
         {'name': 'local_s_patch',
          'shape': tuple([None] + exp.params['local_patch_input_shape'])}),
        (lasagne.layers.Conv2DLayer,
         {'name': 'local_s_patch_conv1',
          'num_filters': exp.params['local_s_patch_conv1_num_filters'],
          'filter_size': exp.params['local_s_patch_conv1_filter_size']}),
        (lasagne.layers.MaxPool2DLayer,
         {'name': 'local_s_patch_pool1',
          'pool_size': exp.params['local_s_patch_pool1_pool_size']}),
        (lasagne.layers.Conv2DLayer,
         {'name': 'local_s_patch_conv2',
          'num_filters': exp.params['local_s_patch_conv2_num_filters'],
          'filter_size': exp.params['local_s_patch_conv2_filter_size']}),
        (lasagne.layers.MaxPool2DLayer,
         {'name': 'local_s_patch_pool2',
          'pool_size': exp.params['local_s_patch_pool2_pool_size']}),
        (lasagne.layers.Conv2DLayer,
         {'name': 'local_s_patch_conv3',
          'num_filters': exp.params['local_s_patch_conv3_num_filters'],
          'filter_size': exp.params['local_s_patch_conv3_filter_size']}),
        (lasagne.layers.MaxPool2DLayer,
         {'name': 'local_s_patch_pool3',
          'pool_size': exp.params['local_s_patch_pool3_pool_size']}),
        (lasagne.layers.FlattenLayer, {'name': 'local_s_patch_flat1'}),

        # Layers for the axial context patch.
        (lasagne.layers.InputLayer,
         {'name': 'context_a_patch',
          'shape': tuple([None] + exp.params['context_patch_input_shape'])}),
        (lasagne.layers.Conv2DLayer,
         {'name': 'context_a_patch_conv1',
          'num_filters': exp.params['context_a_patch_conv1_num_filters'],
          'filter_size': exp.params['context_a_patch_conv1_filter_size']}),
        (lasagne.layers.MaxPool2DLayer,
         {'name': 'context_a_patch_pool1',
          'pool_size': exp.params['context_a_patch_pool1_pool_size']}),
        (lasagne.layers.Conv2DLayer,
         {'name': 'context_a_patch_conv2',
          'num_filters': exp.params['context_a_patch_conv2_num_filters'],
          'filter_size': exp.params['context_a_patch_conv2_filter_size']}),
        (lasagne.layers.MaxPool2DLayer,
         {'name': 'context_a_patch_pool2',
          'pool_size': exp.params['context_a_patch_pool2_pool_size']}),
        (lasagne.layers.Conv2DLayer,
         {'name': 'context_a_patch_conv3',
          'num_filters': exp.params['context_a_patch_conv3_num_filters'],
          'filter_size': exp.params['context_a_patch_conv3_filter_size']}),
        (lasagne.layers.MaxPool2DLayer,
         {'name': 'context_a_patch_pool3',
          'pool_size': exp.params['context_a_patch_pool3_pool_size']}),
        (lasagne.layers.FlattenLayer, {'name': 'context_a_patch_flat1'}),

        # Layers for the coronal context patch.
        (lasagne.layers.InputLayer,
         {'name': 'context_c_patch',
          'shape': tuple([None] + exp.params['context_patch_input_shape'])}),
        (lasagne.layers.Conv2DLayer,
         {'name': 'context_c_patch_conv1',
          'num_filters': exp.params['context_c_patch_conv1_num_filters'],
          'filter_size': exp.params['context_c_patch_conv1_filter_size']}),
        (lasagne.layers.MaxPool2DLayer,
         {'name': 'context_c_patch_pool1',
          'pool_size': exp.params['context_c_patch_pool1_pool_size']}),
        (lasagne.layers.Conv2DLayer,
         {'name': 'context_c_patch_conv2',
          'num_filters': exp.params['context_c_patch_conv2_num_filters'],
          'filter_size': exp.params['context_c_patch_conv2_filter_size']}),
        (lasagne.layers.MaxPool2DLayer,
         {'name': 'context_c_patch_pool2',
          'pool_size': exp.params['context_c_patch_pool2_pool_size']}),
        (lasagne.layers.Conv2DLayer,
         {'name': 'context_c_patch_conv3',
          'num_filters': exp.params['context_c_patch_conv3_num_filters'],
          'filter_size': exp.params['context_c_patch_conv3_filter_size']}),
        (lasagne.layers.MaxPool2DLayer,
         {'name': 'context_c_patch_pool3',
          'pool_size': exp.params['context_c_patch_pool3_pool_size']}),
        (lasagne.layers.FlattenLayer, {'name': 'context_c_patch_flat1'}),

        # Layers for the sagittal context patch.
        (lasagne.layers.InputLayer,
         {'name': 'context_s_patch',
          'shape': tuple([None] + exp.params['context_patch_input_shape'])}),
        (lasagne.layers.Conv2DLayer,
         {'name': 'context_s_patch_conv1',
          'num_filters': exp.params['context_s_patch_conv1_num_filters'],
          'filter_size': exp.params['context_s_patch_conv1_filter_size']}),
        (lasagne.layers.MaxPool2DLayer,
         {'name': 'context_s_patch_pool1',
          'pool_size': exp.params['context_s_patch_pool1_pool_size']}),
        (lasagne.layers.Conv2DLayer,
         {'name': 'context_s_patch_conv2',
          'num_filters': exp.params['context_s_patch_conv2_num_filters'],
          'filter_size': exp.params['context_s_patch_conv2_filter_size']}),
        (lasagne.layers.MaxPool2DLayer,
         {'name': 'context_s_patch_pool2',
          'pool_size': exp.params['context_s_patch_pool2_pool_size']}),
        (lasagne.layers.Conv2DLayer,
         {'name': 'context_s_patch_conv3',
          'num_filters': exp.params['context_s_patch_conv3_num_filters'],
          'filter_size': exp.params['context_s_patch_conv3_filter_size']}),
        (lasagne.layers.MaxPool2DLayer,
         {'name': 'context_s_patch_pool3',
          'pool_size': exp.params['context_s_patch_pool3_pool_size']}),
        (lasagne.layers.FlattenLayer, {'name': 'context_s_patch_flat1'}),

        # Layers for the landmark displacement.
        (lasagne.layers.InputLayer,
         {'name': 'landmark_1', 'shape': (None, 3)}),
        (lasagne.layers.DenseLayer,
         {'name': 'landmark_1_dense1',
          'num_units': exp.params['landmark_1_dense1_num_units']}),
        (lasagne.layers.DenseLayer,
         {'name': 'landmark_1_dense2',
          'num_units': exp.params['landmark_1_dense2_num_units']}),
        (lasagne.layers.DenseLayer,
         {'name': 'landmark_1_dense3',
          'num_units': exp.params['landmark_1_dense3_num_units']}),
        (lasagne.layers.InputLayer,
         {'name': 'landmark_2', 'shape': (None, 3)}),
        (lasagne.layers.DenseLayer,
         {'name': 'landmark_2_dense1',
          'num_units': exp.params['landmark_2_dense1_num_units']}),
        (lasagne.layers.DenseLayer,
         {'name': 'landmark_2_dense2',
          'num_units': exp.params['landmark_2_dense2_num_units']}),
        (lasagne.layers.DenseLayer,
         {'name': 'landmark_2_dense3',
          'num_units': exp.params['landmark_2_dense3_num_units']}),
        (lasagne.layers.InputLayer,
         {'name': 'landmark_3', 'shape': (None, 3)}),
        (lasagne.layers.DenseLayer,
         {'name': 'landmark_3_dense1',
          'num_units': exp.params['landmark_3_dense1_num_units']}),
        (lasagne.layers.DenseLayer,
         {'name': 'landmark_3_dense2',
          'num_units': exp.params['landmark_3_dense2_num_units']}),
        (lasagne.layers.DenseLayer,
         {'name': 'landmark_3_dense3',
          'num_units': exp.params['landmark_3_dense3_num_units']}),

        # Layers for concatenation and output.
        (lasagne.layers.ConcatLayer,
         {'name': 'concat',
          'incomings': ['local_a_patch_flat1', 'local_c_patch_flat1',
                        'local_s_patch_flat1', 'context_a_patch_flat1',
                        'context_c_patch_flat1', 'context_s_patch_flat1',
                        'landmark_1_dense3', 'landmark_2_dense3',
                        'landmark_3_dense3']}),
        (lasagne.layers.DenseLayer,
         {'name': 'join_dense1',
          'num_units': exp.params['join_dense1_num_units']}),
        (lasagne.layers.DenseLayer,
         {'name': 'output', 'num_units': 1,
          'nonlinearity': lasagne.nonlinearities.sigmoid}),

    ],

    # Predict segmentation probabilities.
    regression=True,

    # Loss function.
    objective_loss_function=lasagne.objectives.binary_crossentropy,

    # Optimization method.
    update=lasagne.updates.nesterov_momentum,
    update_learning_rate=exp.params['update_learning_rate'],
    update_momentum=exp.params['update_momentum'],

    # Iteration options.
    max_epochs=exp.params['max_epochs'],
    train_split=nolearn.lasagne.TrainSplit(exp.params['validation_prop']),

    # Other options.
    verbose=1
)
net.initialize()

# Record information to be used for printing training progress.
total_points = 0
for training_map in training_maps:
    total_points += np.count_nonzero(training_map)
elapsed_training_time = 0

# Train the network using a hybrid online/mini-batch approach.
for i, (input_batch, output_batch) in enumerate(ext.iterate_multiple(
        training_vols, training_maps, exp.params['batch_size'])):
    # Train and time the process.
    iteration_start_time = time.time()
    net.fit(input_batch, output_batch)
    elapsed_training_time += time.time() - iteration_start_time

    # Print the expected time remaining.
    pdl.utils.print_progress(elapsed_training_time,
                             (i + 1) * exp.params['batch_size'],
                             total_points)

print("Training complete.\n\n")

# Record results from training.
exp.add_result('training_time', elapsed_training_time)

# Try to pickle the network (which keeps the training history), but if this is
# not possible due to the size of the net then just save the weights.
try:
    exp.pickle_network(net, 'net')
except RuntimeError:
    exp.save_network_weights(net, 'net_weights')

# Perform predictions on all testing volumes in the set.
print('Beginning predictions.\n')
prediction_start_time = time.time()
for i, testing_vol in list(enumerate(testing_vols)):

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
