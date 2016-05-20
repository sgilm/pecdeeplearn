from __future__ import division

import lasagne
import numpy as np
import nolearn.lasagne
import pecdeeplearn as pdl
import data_path
import time


# Create an experiment object to keep track of parameters and facilitate data
# loading and save_allowed.
exp = pdl.utils.Experiment(data_path.get(), 'single_acs_conv_triple_landmark')
exp.add_param('volume_depth', 60)
exp.add_param('max_points_per_volume', 80000)
exp.add_param('margins', (15, 15, 15))
exp.add_param('min_seg_points', 100)
exp.add_param('input_patch_shape', [1, 31, 31])
exp.add_param('axial_patch_shape', [1, 31, 31])
exp.add_param('coronal_patch_shape', [31, 1, 31])
exp.add_param('sagittal_patch_shape', [31, 31, 1])
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

# List and load all vols.
vol_list = exp.list_volumes()
vols = [exp.load_volume(vol) for vol in vol_list]

# Standardise the data.
pdl.utils.standardise_volumes(vols)

# Take a set of slices centred about the left nipple.
centre_slices = [int(vol.landmarks['Left nipple'][0]) for vol in vols]
vols = [vol[(centre_slices[i] - exp.params['volume_depth'] // 2):
            (centre_slices[i] + exp.params['volume_depth'] // 2)]
        for i, vol in enumerate(vols)]

# Discard volumes with little segmentation data.
vols = [vol for vol in vols
        if np.sum(vol.seg_data) > exp.params['min_seg_points']]

# Create training maps.
point_maps = [
    pdl.extraction.half_half_map(
        vol,
        max_points=exp.params['max_points_per_volume'],
        margins=exp.params['margins']
    )
    for vol in vols]

# Create an Extractor.
ext = pdl.extraction.Extractor()

# Add features.
ext.add_feature(
    feature_name='axial_patch',
    feature_function=lambda volume, point:
    pdl.extraction.patch(volume, point, exp.params['axial_patch_shape'])
)
ext.add_feature(
    feature_name='coronal_patch',
    feature_function=lambda volume, point:
    pdl.extraction.patch(volume, point, exp.params['coronal_patch_shape'])
)
ext.add_feature(
    feature_name='sagittal_patch',
    feature_function=lambda volume, point:
    pdl.extraction.patch(volume, point, exp.params['sagittal_patch_shape'])
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
         {'name': 'axial_patch',
          'shape': tuple([None] + exp.params['input_patch_shape'])}),
        (lasagne.layers.Conv2DLayer,
         {'name': 'axial_conv', 'num_filters': exp.params['num_filters'],
          'filter_size': exp.params['filter_size']}),
        (lasagne.layers.DenseLayer,
         {'name': 'axial_patch_dense',
          'num_units': exp.params['patch_num_dense_units']}),

        # Layers for the coronal local patch.
        (lasagne.layers.InputLayer,
         {'name': 'coronal_patch',
          'shape': tuple([None] + exp.params['input_patch_shape'])}),
        (lasagne.layers.Conv2DLayer,
         {'name': 'coronal_conv', 'num_filters': exp.params['num_filters'],
          'filter_size': exp.params['filter_size']}),
        (lasagne.layers.DenseLayer,
         {'name': 'coronal_patch_dense',
          'num_units': exp.params['patch_num_dense_units']}),

        # Layers for the axial local patch.
        (lasagne.layers.InputLayer,
         {'name': 'sagittal_patch',
          'shape': tuple([None] + exp.params['input_patch_shape'])}),
        (lasagne.layers.Conv2DLayer,
         {'name': 'sagittal_conv', 'num_filters': exp.params['num_filters'],
          'filter_size': exp.params['filter_size']}),
        (lasagne.layers.DenseLayer,
         {'name': 'sagittal_patch_dense',
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
          'incomings': ['axial_patch_dense', 'coronal_patch_dense',
                        'sagittal_patch_dense', 'landmark_1_dense',
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

# Record information to be used for printing progress.
total_points = np.count_nonzero(point_maps)
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
exp.pickle_volume(predicted_volume, 'predicted_volume')

# Record the parameters.
exp.record_params()
