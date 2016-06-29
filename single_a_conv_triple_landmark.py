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
exp.create_experiment('single_a_conv_triple_landmark')
exp.add_param('num_training_volumes', 40)
exp.add_param('max_points_per_volume', 50000)
exp.add_param('margins', (20, 20, 0))
exp.add_param('patch_shape', [41, 41, 1])
exp.add_param('input_patch_shape', [1, 41, 41])
exp.add_param('landmark_1', 'Sternal angle')
exp.add_param('landmark_2', 'Left nipple')
exp.add_param('landmark_3', 'Right nipple')
exp.add_param('filter_size', (3, 3))
exp.add_param('num_filters', 64)
exp.add_param('patch_num_dense_units', 500)
exp.add_param('landmark_1_num_dense_units', 500)
exp.add_param('landmark_2_num_dense_units', 500)
exp.add_param('landmark_3_num_dense_units', 500)
exp.add_param('batch_size', 5000)
exp.add_param('update_learning_rate', 0.0001)
exp.add_param('update_momentum', 0.9)
exp.add_param('max_epochs', 100)

# List and load all volumes.
vol_list = exp.list_volumes()
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
    feature_name='patch',
    feature_function=lambda volume, point:
    pdl.extraction.patch(volume, point, exp.params['patch_shape'])
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

        # Layers for the local patch.
        (lasagne.layers.InputLayer,
         {'name': 'patch',
          'shape': tuple([None] + exp.params['input_patch_shape'])}),
        (lasagne.layers.Conv2DLayer,
         {'name': 'conv', 'num_filters': exp.params['num_filters'],
          'filter_size': exp.params['filter_size']}),
        (lasagne.layers.DenseLayer,
         {'name': 'patch_dense',
          'num_units': exp.params['patch_num_dense_units']}),

        # Layers for the landmark displacement.
        (lasagne.layers.InputLayer,
         {'name': 'landmark_1', 'shape': (None, 3)}),
        (lasagne.layers.DenseLayer,
         {'name': 'landmark_1_dense',
          'num_units': exp.params['landmark_1_num_dense_units']}),
        (lasagne.layers.InputLayer,
         {'name': 'landmark_2', 'shape': (None, 3)}),
        (lasagne.layers.DenseLayer,
         {'name': 'landmark_2_dense',
          'num_units': exp.params['landmark_2_num_dense_units']}),
        (lasagne.layers.InputLayer,
         {'name': 'landmark_3', 'shape': (None, 3)}),
        (lasagne.layers.DenseLayer,
         {'name': 'landmark_3_dense',
          'num_units': exp.params['landmark_3_num_dense_units']}),

        # Layers for concatenation and output.
        (lasagne.layers.ConcatLayer,
         {'name': 'concat',
          'incomings': ['patch_dense', 'landmark_1_dense',
                        'landmark_2_dense', 'landmark_3_dense']}),
        (lasagne.layers.DenseLayer,
         {'name': 'output', 'num_units': 2,
          'nonlinearity': lasagne.nonlinearities.softmax}),

    ],

    # Optimization method.
    update=lasagne.updates.nesterov_momentum,
    update_learning_rate=exp.params['update_learning_rate'],
    update_momentum=exp.params['update_momentum'],

    # Other options.
    max_epochs=exp.params['max_epochs'],
    verbose=1
)
net.initialize()

# Record information to be used for printing progress.
total_points = np.count_nonzero(training_maps)
start_time = time.time()

# Iterate through and train.
for i, (input_batch, output_batch) in \
        enumerate(ext.iterate_multiple(training_vols, training_maps,
                                       exp.params['batch_size'])):
    net.fit(input_batch, output_batch)
    pdl.utils.print_progress(time.time() - start_time,
                             (i + 1) * exp.params['batch_size'],
                             total_points)
print("Training complete.")

# Record the time taken for training.
exp.add_result('training_time', time.time() - start_time)

# Save the network.
exp.save_network(net, 'net')

# Perform predictions.
for testing_vol in testing_vols[:3]:

    # Get the current volume for testing and perform the prediction.
    print("Predicting on volume " + testing_vol.name + ".")
    predicted_vol = ext.predict(net, testing_vol, exp.params['batch_size'])

    # Calculate the Dice coefficient for this prediction and record it.
    dice = pdl.utils.dice_coefficient(testing_vol.seg_data,
                                      predicted_vol.seg_data,
                                      margins=exp.params['margins'])
    exp.add_result(testing_vol.name + '_dice', dice)

    # Save the predictions for comparison.
    exp.pickle_volume(predicted_vol)
    exp.export_nii(predicted_vol)

# Record the parameters
exp.record()
