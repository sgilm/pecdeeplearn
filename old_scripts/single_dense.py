from __future__ import division

import lasagne
import numpy as np
import nolearn.lasagne
import pecdeeplearn as pdl
import data_path
import time


# Create an experiment object to keep track of parameters and facilitate data
# loading and save_allowed.
exp = pdl.utils.Experiment(data_path.get(), 'single_dense')
exp.add_param('min_seg_points', 100)
exp.add_param('point_offset', [0, 0, 0])
exp.add_param('batch_size', 5000)
exp.add_param('num_hidden_units', 2000)
exp.add_param('update_learning_rate', 0.0001)
exp.add_param('update_momentum', 0.9)
exp.add_param('max_epochs', 100)

# List and load all vols.
vol_list = exp.list_volumes()
vols = [exp.load_volume(vol) for vol in vol_list]

# Standardise the data.
pdl.utils.standardise_volumes(vols)

# Take a slice corresponding to the location of the left nipple.
vols = [vol[int(vol.landmarks['Left nipple'][0])] for vol in vols]

# Strip away vols with little segmentation data.
vols = [vol for vol in vols
        if np.sum(vol.seg_data) > exp.params['min_seg_points']]

# Create training maps.
point_maps = [pdl.extraction.half_half_map(vol) for vol in vols]

# Create an Extractor.
ext = pdl.extraction.Extractor()

# Add features.
ext.add_feature(
    feature_name='point',
    feature_function=lambda volume, point:
    pdl.extraction.point_offset(volume, point, exp.params['point_offset'])
)

# Create net.
net = nolearn.lasagne.NeuralNet(
    layers=[

        # Three layers; one hidden layer.
        (lasagne.layers.InputLayer,
         {'name': 'point', 'shape': (None, 1)}),
        (lasagne.layers.DenseLayer,
         {'name': 'hidden', 'num_units': exp.params['num_hidden_units']}),
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

# Record information to be used for printing progress.
total_points = np.sum(point_maps)
start_time = time.time()

# Iterate through and train.
for i, (input_batch, output_batch) in \
        enumerate(ext.iterate_multiple(vols[:-1],
                                       point_maps[:-1],
                                       exp.params['batch_size'])):
    net.fit(input_batch, output_batch)
    pdl.utils.print_progress(start_time,
                             (i + 1) * exp.params['batch_size'],
                             total_points)
print("Training complete.")

# Save the network.
exp.save_network(net, 'net')

# Predict on the last volume.
test_volume = vols[-1]
predicted_volume = ext.predict(net, test_volume, exp.params['batch_size'])

# Save the volumes for comparison.
exp.pickle_volume(test_volume, 'test_volume')
exp.pickle_volume(predicted_volume, 'predicted_vol')

# Record the parameters
exp.record_params()
