import lasagne
import numpy as np
import nolearn.lasagne
import nolearn.lasagne.visualize
import pecdeeplearn as pdl


def first():

    # List and load all volumes, then switch them to the axial orientation.
    volume_list = pdl.utils.list_volumes()
    volumes = [pdl.utils.load_volume(volume) for volume in volume_list]
    for volume in volumes:
        volume.switch_orientation('acs')

    # Take a slice corresponding to the location of the left nipple.
    volumes = [volume[int(volume.landmarks['Left nipple'][0])]
               for volume in volumes]

    # Create an Extractor.
    ext = pdl.extraction.Extractor()

    # Add features.
    kernel_shape = [1, 13, 13]
    ext.add_feature(
        feature_name='patch',
        feature_function=lambda volume, point:
        pdl.extraction.patch(volume, point, kernel_shape)
    )

    # Create net.
    net = nolearn.lasagne.NeuralNet(
        layers=[

            # Three layers; one hidden layer.
            ('input', lasagne.layers.InputLayer),
            ('hidden', lasagne.layers.DenseLayer),
            ('output', lasagne.layers.DenseLayer),
        ],

        # Layer parameters.
        input_shape=(None, 169),  # 13x13 input voxels per batch.
        hidden_num_units=100,
        output_nonlinearity=lasagne.nonlinearities.sigmoid,
        output_num_units=2,

        # Optimization method.
        update=lasagne.updates.nesterov_momentum,
        update_learning_rate=0.001,
        update_momentum=0.9,

        max_epochs=2,
        verbose=1,
    )

    # Create map and define batch size.
    seg_map = pdl.extraction.segmentation_map(volumes)
    batch_size = 100

    # Iterate through and train.
    for volume in volumes[0:-2]:
        for input_batch, output_batch, _ in \
                ext.iterate_single(volume, seg_map, batch_size):
            net.fit(input_batch.reshape(batch_size, 169), output_batch)

    # Predict on second to last volumes.
    test_volume = volumes[-2]
    predicted_volume = ext.predict(net, test_volume, batch_size)

    # Compare volumes visually.
    test_volume.show_slice(0)
    predicted_volume.show_slice(0)


def second():

    # List and load all volumes, then switch them to the axial orientation.
    volume_list = pdl.utils.list_volumes()
    volumes = [pdl.utils.load_volume(volume) for volume in volume_list]
    for volume in volumes:
        volume.switch_orientation('acs')

    # Take a slice corresponding to the location of the left nipple.
    volumes = [volume[int(volume.landmarks['Left nipple'][0])]
               for volume in volumes]

    # Create an Extractor.
    ext = pdl.extraction.Extractor()

    # Add features.
    kernel_shape = [1, 21, 21]
    ext.add_feature(
        feature_name='patch',
        feature_function=lambda volume, point:
        pdl.extraction.patch(volume, point, kernel_shape)
    )

    # Create net.
    net = nolearn.lasagne.NeuralNet(
        layers=[

            (lasagne.layers.InputLayer,
             {'name': 'patch',
              'shape': (None, 1, 21, 21)}),
            (lasagne.layers.Conv2DLayer,
             {'name': 'conv1',
              'num_filters': 200,
              'filter_size': (5, 5)}),
            (lasagne.layers.MaxPool2DLayer,
             {'name': 'pool1',
              'pool_size': (2, 2)}),
            (lasagne.layers.Conv2DLayer,
             {'name': 'conv2',
              'num_filters': 400,
              'filter_size': (3, 3)}),
            (lasagne.layers.MaxPool2DLayer,
             {'name': 'pool2',
              'pool_size': (2, 2)}),
            (lasagne.layers.DenseLayer,
             {'name': 'output',
              'num_units': 2,
              'nonlinearity': lasagne.nonlinearities.softmax}),

        ],

        update=lasagne.updates.nesterov_momentum,
        update_learning_rate=0.0001,
        update_momentum=0.9,

        max_epochs=10,
        verbose=True,
    )

    # Create probability bins (for later creating training maps).
    bins, prob_bins = pdl.extraction.probability_bins(volumes, scale=0.75)

    # Define batch size.
    batch_size = 1000

    # Define a factor to ensure training batches are balanced.
    factor = 0.2

    # Iterate through and train, making sure the batch is balanced.
    for volume in volumes[0:-2]:
        prob_map = pdl.extraction.probability_map(volume, bins, prob_bins)
        for input_batch, output_batch, _ in \
                ext.iterate_single(volume, prob_map, batch_size):
            num_positives = np.count_nonzero(output_batch)
            min_positives = batch_size * factor
            max_positives = batch_size * (1 - factor)
            if min_positives < num_positives < max_positives:
                net.fit(input_batch, output_batch)
    print('Finished training.')

    # Test on the second to last volumes.
    test_volume = volumes[-2]

    print('Performing segmentation...')
    predicted_volume = ext.predict(net, test_volume, batch_size)
    print('Segmentation complete.')

    # Save predicted volumes.
    pdl.utils.pickle_volume(predicted_volume, 'second.pkl')

    # Compare segmentations.
    test_volume.show_slice(0)
    predicted_volume.show_slice(0)


def third(train=True):

    # List and load all volumes, then switch them to the axial orientation.
    volume_list = pdl.utils.list_volumes()
    volumes = [pdl.utils.load_volume(volume) for volume in volume_list]
    for volume in volumes:
        volume.switch_orientation('acs')

    # Take a slice corresponding to the location of the left nipple.
    volumes = [volume[int(volume.landmarks['Left nipple'][0])]
               for volume in volumes]

    # Create an Extractor.
    ext = pdl.extraction.Extractor()

    # Add features.
    ext.add_feature(
        feature_name='local_patch',
        feature_function=lambda volume, point:
        pdl.extraction.patch(volume, point, [1, 25, 25])
    )
    ext.add_feature(
        feature_name='context_patch',
        feature_function=lambda volume, point:
        pdl.extraction.scaled_patch(volume, point, [1, 50, 50], [1, 25, 25])
    )
    ext.add_feature(
        feature_name='sternal_angle',
        feature_function=lambda volume, point:
        pdl.extraction.landmark_displacement(volume, point, 'Sternal angle')
    )
    ext.add_feature(
        feature_name='left_nipple',
        feature_function=lambda volume, point:
        pdl.extraction.landmark_displacement(volume, point, 'Left nipple')
    )
    ext.add_feature(
        feature_name='right_nipple',
        feature_function=lambda volume, point:
        pdl.extraction.landmark_displacement(volume, point, 'Right nipple')
    )

    # Create the net.
    net = nolearn.lasagne.NeuralNet(
        layers=[

            # Layers for the local patch.
            (lasagne.layers.InputLayer,
             {'name': 'local_patch', 'shape': (None, 1, 25, 25)}),
            (lasagne.layers.Conv2DLayer,
             {'name': 'local_patch_conv1', 'num_filters': 150,
              'filter_size': (5, 5)}),
            (lasagne.layers.MaxPool2DLayer,
             {'name': 'local_patch_pool1', 'pool_size': (2, 2)}),
            (lasagne.layers.Conv2DLayer,
             {'name': 'local_patch_conv2', 'num_filters': 150,
              'filter_size': (3, 3)}),
            (lasagne.layers.MaxPool2DLayer,
             {'name': 'local_patch_pool2', 'pool_size': (2, 2)}),
            (lasagne.layers.DenseLayer,
             {'name': 'local_patch_dense1', 'num_units': 150}),

            # Layers for the context patch.
            (lasagne.layers.InputLayer,
             {'name': 'context_patch', 'shape': (None, 1, 25, 25)}),
            (lasagne.layers.Conv2DLayer,
             {'name': 'context_patch_conv1', 'num_filters': 150,
              'filter_size': (5, 5)}),
            (lasagne.layers.MaxPool2DLayer,
             {'name': 'context_patch_pool1', 'pool_size': (2, 2)}),
            (lasagne.layers.Conv2DLayer,
             {'name': 'context_patch_conv2', 'num_filters': 150,
              'filter_size': (3, 3)}),
            (lasagne.layers.MaxPool2DLayer,
             {'name': 'context_patch_pool2', 'pool_size': (2, 2)}),
            (lasagne.layers.DenseLayer,
             {'name': 'context_patch_dense1', 'num_units': 150}),

            # Layers for the landmark displacements.
            (lasagne.layers.InputLayer,
             {'name': 'sternal_angle', 'shape': (None, 3)}),
            (lasagne.layers.DenseLayer,
             {'name': 'sternal_angle_dense1', 'num_units': 75}),
            (lasagne.layers.InputLayer,
             {'name': 'left_nipple', 'shape': (None, 3)}),
            (lasagne.layers.DenseLayer,
             {'name': 'left_nipple_dense1', 'num_units': 75}),
            (lasagne.layers.InputLayer,
             {'name': 'right_nipple', 'shape': (None, 3)}),
            (lasagne.layers.DenseLayer,
             {'name': 'right_nipple_dense1', 'num_units': 75}),

            # Layers for concatenation and output.
            (lasagne.layers.ConcatLayer,
             {'incomings': ['local_patch_dense1', 'context_patch_dense1',
                            'sternal_angle_dense1', 'left_nipple_dense1',
                            'right_nipple_dense1']}),
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

    # Define the batch size.
    batch_size = 1000

    if train:

        # Train on all but the last two volumes, and use a half-half map.
        for index, volume in enumerate(volumes[0:-2]):
            for input_batch, output_batch, _ in ext.iterate_single(
                    volume, pdl.extraction.half_half_map(volume), batch_size):
                net.fit(input_batch, output_batch)
            print('\nFinished training on volumes #' + str(index) + '.\n')

        print('Finished training.')

        # Save the network for later use.
        pdl.utils.pickle_network(net, 'third.pkl')

    else:

        # Load the net for predictions.
        pdl.utils.unpickle_network('third.pkl')

    # Test on the reserved second to last volumes.
    test_volume = volumes[-2]

    # Perform the prediction.
    print('Performing test segmentation.')
    predicted_volume = ext.predict(net, test_volume, batch_size)
    print('Segmentation complete.')

    # Save predicted volumes for analysis.
    pdl.utils.pickle_volume(predicted_volume, 'third.pkl')
    test_volume.show_slice(0)
    predicted_volume.show_slice(0)


def fourth(train=True):

    # List and load all volumes, then switch them to the axial orientation.
    volume_list = pdl.utils.list_volumes()
    volumes = [pdl.utils.load_volume(volume) for volume in volume_list]
    for volume in volumes:
        volume.switch_orientation('acs')

    # Take a slice corresponding to the location of the left nipple.
    volumes = [volume[int(volume.landmarks['Left nipple'][0])]
               for volume in volumes]

    # Strip away volumes with little segmentation data.
    min_seg_points = 100
    volumes = [volume for volume in volumes
               if np.sum(volume.seg_data) > min_seg_points]

    # Create training maps.
    point_maps = [pdl.extraction.half_half_map(volume) for volume in volumes]

    # Create an Extractor.
    ext = pdl.extraction.Extractor()

    # Add features.
    ext.add_feature(
        feature_name='local_patch',
        feature_function=lambda volume, point:
        pdl.extraction.patch(volume, point, [1, 30, 30])
    )
    ext.add_feature(
        feature_name='context_patch',
        feature_function=lambda volume, point:
        pdl.extraction.scaled_patch(volume, point, [1, 75, 75], [1, 30, 30])
    )
    ext.add_feature(
        feature_name='sternal_angle',
        feature_function=lambda volume, point:
        pdl.extraction.landmark_displacement(volume, point, 'Sternal angle')
    )
    ext.add_feature(
        feature_name='left_nipple',
        feature_function=lambda volume, point:
        pdl.extraction.landmark_displacement(volume, point, 'Left nipple')
    )
    ext.add_feature(
        feature_name='right_nipple',
        feature_function=lambda volume, point:
        pdl.extraction.landmark_displacement(volume, point, 'Right nipple')
    )

    # Create the net.
    net = nolearn.lasagne.NeuralNet(
        layers=[

            # Layers for the local patch.
            (lasagne.layers.InputLayer,
             {'name': 'local_patch', 'shape': (None, 1, 30, 30)}),
            (lasagne.layers.Conv2DLayer,
             {'name': 'local_patch_conv1', 'num_filters': 32,
              'filter_size': (10, 10)}),
            (lasagne.layers.MaxPool2DLayer,
             {'name': 'local_patch_pool1', 'pool_size': (2, 2)}),
            (lasagne.layers.Conv2DLayer,
             {'name': 'local_patch_conv2', 'num_filters': 64,
              'filter_size': (5, 5)}),
            (lasagne.layers.MaxPool2DLayer,
             {'name': 'local_patch_pool2', 'pool_size': (2, 2)}),
            (lasagne.layers.DenseLayer,
             {'name': 'local_patch_dense1', 'num_units': 500}),

            # Layers for the context patch.
            (lasagne.layers.InputLayer,
             {'name': 'context_patch', 'shape': (None, 1, 30, 30)}),
            (lasagne.layers.Conv2DLayer,
             {'name': 'context_patch_conv1', 'num_filters': 32,
              'filter_size': (10, 10)}),
            (lasagne.layers.MaxPool2DLayer,
             {'name': 'context_patch_pool1', 'pool_size': (2, 2)}),
            (lasagne.layers.Conv2DLayer,
             {'name': 'context_patch_conv2', 'num_filters': 64,
              'filter_size': (5, 5)}),
            (lasagne.layers.MaxPool2DLayer,
             {'name': 'context_patch_pool2', 'pool_size': (2, 2)}),
            (lasagne.layers.DenseLayer,
             {'name': 'context_patch_dense1', 'num_units': 500}),

            # Layers for the landmark displacements.
            (lasagne.layers.InputLayer,
             {'name': 'sternal_angle', 'shape': (None, 3)}),
            (lasagne.layers.DenseLayer,
             {'name': 'sternal_angle_dense1', 'num_units': 100}),
            (lasagne.layers.InputLayer,
             {'name': 'left_nipple', 'shape': (None, 3)}),
            (lasagne.layers.DenseLayer,
             {'name': 'left_nipple_dense1', 'num_units': 100}),
            (lasagne.layers.InputLayer,
             {'name': 'right_nipple', 'shape': (None, 3)}),
            (lasagne.layers.DenseLayer,
             {'name': 'right_nipple_dense1', 'num_units': 100}),

            # Layers for concatenation and output.
            (lasagne.layers.ConcatLayer,
             {'incomings': ['local_patch_dense1', 'context_patch_dense1',
                            'sternal_angle_dense1', 'left_nipple_dense1',
                            'right_nipple_dense1']}),
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

    # Define the batch size.
    batch_size = 10000

    if train:

        # Train on all but the last two volumes, and use a half-half map.
        for input_batch, output_batch in ext.iterate_multiple(
                volumes[:-1], point_maps[:-1], batch_size):
            net.fit(input_batch, output_batch)

        print('Finished training.')

        # Plot training losses.
        nolearn.lasagne.visualize.plot_loss(net).show()

        # Save the net for later use.
        pdl.utils.pickle_network(net, 'fourth.pkl')

    else:

        # Load the net for predictions.
        pdl.utils.unpickle_network('fourth.pkl')

    # Plot convolutional layer weights.
    nolearn.lasagne.visualize.plot_conv_weights(
        net.layers_['local_patch_conv1']
    ).show()

    # Test on the reserved last volume.
    test_volume = volumes[-1]

    # Perform the prediction.
    print('Performing test segmentation.')
    predicted_volume = ext.predict(net, test_volume, batch_size)
    print('Segmentation complete.')

    # Save predicted volumes for analysis, and compare visually.
    pdl.utils.pickle_volume(predicted_volume, 'fourth.pkl')
    test_volume.show_slice(0)
    predicted_volume.show_slice(0)
