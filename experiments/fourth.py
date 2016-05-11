from __future__ import division

import lasagne
import numpy as np
import nolearn.lasagne
import pecdeeplearn as pdl


# List and load all vols, then switch them to the axial orientation.
vol_list = pdl.utils.list_volumes()
vols = [pdl.utils.load_volume(vol) for vol in vol_list]
for vol in vols:
    vol.switch_orientation('acs')

# Take a slice corresponding to the location of the left nipple.
vols = [vol[int(vol.landmarks['Left nipple'][0])] for vol in vols]

# Strip away vols with little segmentation data.
min_seg_points = 100
vols = [vol for vol in vols if np.sum(vol.seg_data) > min_seg_points]

# Create training maps.
point_maps = [pdl.extraction.half_half_map(vol) for vol in vols]

# Create an Extractor.
ext = pdl.extraction.Extractor()

# Add features.
ext.add_feature(
    feature_name='patch',
    feature_function=lambda volume, point:
    pdl.extraction.patch(volume, point, [1, 21, 21])
)
ext.add_feature(
    feature_name='voxel_intensity',
    feature_function=lambda volume, point:
    pdl.extraction.point_offset(volume, point, [0, 0, 0])
)
ext.add_feature(
    feature_name='sternal_angle',
    feature_function=lambda volume, point:
    pdl.extraction.landmark_displacement(volume, point, 'Sternal angle')
)

# Define batch size.
batch_size = 5000

# Create the net.
net = nolearn.lasagne.NeuralNet(
    layers=[

        # Layers for the local patch.
        (lasagne.layers.InputLayer,
         {'name': 'patch', 'shape': (None, 1, 21, 21)}),
        (lasagne.layers.Conv2DLayer,
         {'name': 'patch_conv', 'num_filters': 64,
          'filter_size': (3, 3)}),
        (lasagne.layers.DenseLayer,
         {'name': 'patch_dense', 'num_units': 100}),

        # Layers for the voxel intensity.
        (lasagne.layers.InputLayer,
         {'name': 'voxel_intensity', 'shape': (None, 1)}),
        (lasagne.layers.DenseLayer,
         {'name': 'voxel_intensity_dense', 'num_units': 2000}),

        # Layers for the landmark displacement.
        (lasagne.layers.InputLayer,
         {'name': 'sternal_angle', 'shape': (None, 3)}),
        (lasagne.layers.DenseLayer,
         {'name': 'sternal_angle_dense', 'num_units': 100}),

        # Layers for concatenation and output.
        (lasagne.layers.ConcatLayer,
         {'incomings': ['patch_dense', 'voxel_intensity_dense',
                        'sternal_angle_dense']}),
        (lasagne.layers.DenseLayer,
         {'name': 'output', 'num_units': 2,
          'nonlinearity': lasagne.nonlinearities.softmax}),

    ],

    # Define learning parameters.
    update=lasagne.updates.nesterov_momentum,
    update_learning_rate=0.0001,
    update_momentum=0.9,

    # Define training parameters.
    max_epochs=50,
    verbose=True
)

# Iterate through and train.
for input_batch, output_batch, in \
        ext.iterate_multiple(vols[:-1], point_maps[:-1], batch_size):
    net.fit(input_batch, output_batch)

# Predict on the last volume.
test_volume = vols[-1]
predicted_volume = ext.predict(net, test_volume, batch_size)

# Compare volumes visually.
test_volume.show_slice(0)
predicted_volume.show_slice(0)
