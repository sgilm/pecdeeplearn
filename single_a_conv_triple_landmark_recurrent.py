from __future__ import division

import lasagne
import nolearn.lasagne
import pecdeeplearn as pdl
import data_path
import time
import numpy as np


# Create an experiment object to keep track of parameters and facilitate data
# loading and saving.
exp = pdl.utils.Experiment(data_path.get())
exp.create_experiment('double_a_conv_triple_landmark_recurrent')
exp.add_param('num_training_volumes', 2)
exp.add_param('max_points_per_volume', 25000)
exp.add_param('margins', (12, 12, 0))
exp.add_param('local_patch_shape', [25, 25, 1])
exp.add_param('local_patch_input_shape', [1, 25, 25])
exp.add_param('local_patch_conv1_filter_size', (5, 5))
exp.add_param('local_patch_conv1_num_filters', 64)
exp.add_param('local_patch_pool1_pool_size', (2, 2))
exp.add_param('local_patch_recurrent_num_steps', 2)
exp.add_param('landmark_1', 'Sternal angle')
exp.add_param('landmark_2', 'Left nipple')
exp.add_param('landmark_3', 'Right nipple')
exp.add_param('landmark_1_dense_num_units', 1000)
exp.add_param('landmark_2_dense_num_units', 1000)
exp.add_param('landmark_3_dense_num_units', 1000)
exp.add_param('join_dense_num_units', 1000)
exp.add_param('batch_size', 5000)
exp.add_param('update_learning_rate', 0.00001)
exp.add_param('update_momentum', 0.9)
exp.add_param('max_epochs', 100)
exp.add_param('validation_prop', 0.2)
exp.add_param('prediction_margins', (25, 25, 25))

# List and load all volumes.
vol_list = exp.list_volumes()[:4]
test_vol_names = ['VL00080', 'VL00093', 'VL00028', 'VL00077', 'VL00094',
                  'VL00057', 'VL00024', 'VL00066', 'VL00063', 'VL00062',
                  'VL00075', 'VL00069', 'VL00038', 'VL00058', 'VL00031']
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
    pdl.extraction.half_half_map(
        vol,
        max_points=exp.params['max_points_per_volume'],
        margins=exp.params['margins']
    )
    for vol in training_vols]

# Create an Extractor.
ext = pdl.extraction.Extractor()

# Add features.
ext.add_feature(
    feature_name='local_patch',
    feature_function=lambda volume, point:
    pdl.extraction.patch(volume, point, exp.params['local_patch_shape'])
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

# Initialise recurrent architecture.
in_to_hid_incoming = lasagne.layers.InputLayer(
    name='in_to_hid_incoming',
    shape=tuple([None] + exp.params['local_patch_input_shape'])
)
hid_to_hid_incoming = lasagne.layers.InputLayer(
    name='hid_to_hid_incoming',
    shape=tuple([None, exp.params['local_patch_conv1_num_filters']] +
                list(np.array(exp.params['local_patch_input_shape'][1:])))
)
in_to_hid_conv1 = lasagne.layers.Conv2DLayer(
    name='in_to_hid_conv1',
    incoming=in_to_hid_incoming,
    num_filters=exp.params['local_patch_conv1_num_filters'],
    filter_size=exp.params['local_patch_conv1_filter_size'],
    pad='same'
)
hid_to_hid_conv1 = lasagne.layers.Conv2DLayer(
    name='hid_to_hid_conv1',
    incoming=hid_to_hid_incoming,
    num_filters=exp.params['local_patch_conv1_num_filters'],
    filter_size=exp.params['local_patch_conv1_filter_size'],
    pad='same'
)

# Create the net.
net = nolearn.lasagne.NeuralNet(
    layers=[

        # Layers for the local patch.
        (lasagne.layers.InputLayer,
         {'name': 'local_patch',
          'shape': tuple(
              [None, exp.params['local_patch_recurrent_num_steps']] +
              exp.params['local_patch_input_shape'])}
         ),
        (lasagne.layers.CustomRecurrentLayer,
         {'name': 'recurrent', 'incoming': 'local_patch',
          'input_to_hidden': in_to_hid_conv1,
          'hidden_to_hidden': hid_to_hid_conv1, 'only_return_final': True}),
        (lasagne.layers.MaxPool2DLayer,
         {'name': 'local_patch_pool1',
          'pool_size': exp.params['local_patch_pool1_pool_size']}),
        (lasagne.layers.FlattenLayer,
         {'name': 'local_patch_flat'}),

        # Layers for the landmark displacement.
        (lasagne.layers.InputLayer,
         {'name': 'landmark_1', 'shape': (None, 3)}),
        (lasagne.layers.DenseLayer,
         {'name': 'landmark_1_dense',
          'num_units': exp.params['landmark_1_dense_num_units']}),
        (lasagne.layers.InputLayer,
         {'name': 'landmark_2', 'shape': (None, 3)}),
        (lasagne.layers.DenseLayer,
         {'name': 'landmark_2_dense',
          'num_units': exp.params['landmark_2_dense_num_units']}),
        (lasagne.layers.InputLayer,
         {'name': 'landmark_3', 'shape': (None, 3)}),
        (lasagne.layers.DenseLayer,
         {'name': 'landmark_3_dense',
          'num_units': exp.params['landmark_3_dense_num_units']}),

        # Layers for concatenation and output.
        (lasagne.layers.ConcatLayer,
         {'name': 'concat',
          'incomings': ['local_patch_flat', 'landmark_1_dense',
                        'landmark_2_dense', 'landmark_3_dense']}),
        (lasagne.layers.DenseLayer,
         {'name': 'join_dense',
          'num_units': exp.params['join_dense_num_units']}),
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

    input_batch['local_patch'] = \
        np.expand_dims(input_batch['local_patch'], axis=1).repeat(
            exp.params['local_patch_recurrent_num_steps'], 1)

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

    # Perform the prediction on the current testing volume.
    print("Predicting on volume " + testing_vol.name + ".")
    predicted_vol = ext.predict(
        net,
        testing_vol,
        exp.params['batch_size'],
        bounds=testing_vol.bounding_box(
            margins=exp.params['prediction_margins'])
    )

    # Save the prediction probabilities for comparison.
    predicted_name = predicted_vol.name
    predicted_vol.name += "_prob"
    exp.export_nii(predicted_vol)

    # Save the rounded segmentation.
    predicted_vol.name = predicted_name
    predicted_vol.seg_data = np.around(predicted_vol.seg_data).astype('int16')
    exp.export_nii(predicted_vol)

    # Calculate statistics for this prediction and record them.
    correct_positives, false_positives, false_negatives = \
        pdl.utils.prediction_stats(testing_vol.seg_data,
                                   predicted_vol.seg_data)
    dice = pdl.utils.dice_coefficient(testing_vol.seg_data,
                                      predicted_vol.seg_data)
    exp.add_result(testing_vol.name + '_correct_positives', correct_positives)
    exp.add_result(testing_vol.name + '_false_positives', false_positives)
    exp.add_result(testing_vol.name + '_false_negatives', false_negatives)
    exp.add_result(testing_vol.name + '_dice', dice)

    # Print prediction progress.
    pdl.utils.print_progress(time.time() - prediction_start_time,
                             i + 1,
                             len(testing_vols))

# Record the parameters and results.
exp.record()
