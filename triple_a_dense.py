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
exp.create_experiment('triple_a_dense')
exp.add_param('num_training_volumes', 45)
exp.add_param('max_points_per_volume', 50000)
exp.add_param('margins', (12, 12, 0))
exp.add_param('local_patch_shape', [25, 25, 1])
exp.add_param('local_patch_input_shape', [25 * 25])
exp.add_param('local_patch_dense1_num_units', 1000)
exp.add_param('local_patch_dense2_num_units', 1000)
exp.add_param('local_patch_dense3_num_units', 1000)
exp.add_param('batch_size', 1000)
exp.add_param('update_learning_rate', 0.0001)
exp.add_param('update_momentum', 0.9)
exp.add_param('max_epochs', 100)
exp.add_param('validation_prop', 0.2)

# List and load all volumes.
vol_list = exp.list_volumes()
for vol_name in ['VL00033', 'VL00034']:
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
    feature_name='local_patch',
    feature_function=lambda volume, point:
    pdl.extraction.flat_patch(volume, point, exp.params['local_patch_shape'])
)

# Create the net.
net = nolearn.lasagne.NeuralNet(
    layers=[

        # Layers for the local patch.
        (lasagne.layers.InputLayer,
         {'name': 'local_patch',
          'shape': tuple([None] + exp.params['local_patch_input_shape'])}),
        (lasagne.layers.DenseLayer,
         {'name': 'local_patch_dense1',
          'num_units': exp.params['local_patch_dense1_num_units']}),
        (lasagne.layers.DenseLayer,
         {'name': 'local_patch_dense2',
          'num_units': exp.params['local_patch_dense2_num_units']}),
        (lasagne.layers.DenseLayer,
         {'name': 'local_patch_dense3',
          'num_units': exp.params['local_patch_dense3_num_units']}),

        # Layer for output.
        (lasagne.layers.DenseLayer,
         {'name': 'output', 'num_units': 2,
          'nonlinearity': lasagne.nonlinearities.softmax}),

    ],

    # Loss function.
    objective_loss_function=lasagne.objectives.categorical_crossentropy,

    # Optimization method.
    update=lasagne.updates.nesterov_momentum,
    update_learning_rate=exp.params['update_learning_rate'],
    update_momentum=exp.params['update_momentum'],

    # Iteration options.
    max_epochs=1,
    batch_iterator_train=nolearn.lasagne.BatchIterator(
        exp.params['batch_size'], shuffle=True),
    batch_iterator_test=nolearn.lasagne.BatchIterator(
        exp.params['batch_size'], shuffle=True),
    train_split=nolearn.lasagne.TrainSplit(exp.params['validation_prop']),

    # Other options.
    verbose=1
)
net.initialize()

# Record information to be used for printing training progress.
training_start_time = time.time()

# Iterate through and train, recording losses.
training_losses = []
validation_losses = []
for epoch in range(exp.params['max_epochs']):

    # Record the training and validation losses from each epoch to check for
    # convergence.
    epoch_training_loss = 0
    epoch_validation_loss = 0

    # Train the network on a complete sweep of the data.
    for input_batch, output_batch in ext.iterate_multiple(
            training_vols, training_maps, exp.params['batch_size']):
        net.fit(input_batch, output_batch)
        epoch_training_loss += net.train_history_[-1]['train_loss']
        epoch_validation_loss += net.train_history_[-1]['valid_loss']

    # Record results of epoch.
    training_losses.append(epoch_training_loss)
    validation_losses.append(epoch_validation_loss)

    # Print the expected time remaining.
    pdl.utils.print_progress(time.time() - training_start_time,
                             epoch + 1,
                             exp.params['max_epochs'])

print("Training complete.\n\n")

# Record results from training.
exp.add_result('training_time', time.time() - training_start_time)
exp.add_result('training_losses', training_losses)
exp.add_result('validation_losses', validation_losses)

# Try to pickle the network (which keeps the training history), but if this is
# not possible due to the size of the net then just save the weights.
try:
    exp.pickle_network(net, 'net')
except RuntimeError:
    exp.save_network_weights(net, 'net_weights')

# Perform predictions on all testing volumes in the set.
print('Beginning predictions.\n')
prediction_start_time = time.time()
for i, testing_vol in enumerate(testing_vols):

    # Perform the prediction on the current testing volume.
    print("Predicting on volume " + testing_vol.name + ".")
    predicted_vol = ext.predict(
        net,
        testing_vol,
        exp.params['batch_size'],
        bounds=[
            exp.params['margins'],
            np.array(testing_vol.shape) - 1 - np.array(exp.params['margins'])
        ]
    )

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

    # Save the prediction for comparison.
    exp.export_nii(predicted_vol)

    # Print prediction progress.
    pdl.utils.print_progress(time.time() - prediction_start_time,
                             i + 1,
                             len(testing_vols))

# Record the parameters and results.
exp.record()
