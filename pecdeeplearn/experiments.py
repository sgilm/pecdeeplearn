import lasagne
import volumetools
import features
import numpy as np
import maps
import nolearn.lasagne
import datapath
import os


def first():

    # List and load all volumes, then switch them to the axial orientation.
    volume_list = volumetools.list_volumes()
    volumes = [volumetools.load_volume(volume) for volume in volume_list]
    for volume in volumes:
        volume.switch_plane('axial')

    # Take a slice corresponding to the location of the left nipple.
    volumes = [volume[int(volume.landmarks['Left nipple'][0])]
               for volume in volumes]

    # Create an Extractor.
    ext = volumetools.Extractor()

    # Add features.
    kernel_shape = [1, 13, 13]
    ext.add_feature(
        feature_name='patch',
        feature_function=lambda volume, point:
            features.patch(volume, point, kernel_shape)
    )

    # Create net.
    net = nolearn.lasagne.NeuralNet(
        layers = [

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
    seg_map = maps.segmentation_map(volumes)
    batch_size = 100

    # Iterate through and train.
    for volume in volumes[0:-2]:
        ext.set_volume(volume)
        for input_batch, output_batch, _ in ext.iterate(batch_size,
                                                        point_map=seg_map):
            net.fit(input_batch.reshape(batch_size, 169), output_batch)

    # Predict on second to last volume.
    ext.set_volume(volumes[-2])
    test_volume = ext.predict(net)


def second():

    # List and load all volumes, then switch them to the axial orientation.
    volume_list = volumetools.list_volumes()
    volumes = [volumetools.load_volume(volume) for volume in volume_list]
    for volume in volumes:
        volume.switch_plane('axial')

    # Take a slice corresponding to the location of the left nipple.
    volumes = [volume[int(volume.landmarks['Left nipple'][0])]
               for volume in volumes]

    # Create an Extractor.
    ext = volumetools.Extractor()

    # Add features.
    kernel_shape = [1, 21, 21]
    ext.add_feature(
        feature_name='patch',
        feature_function=lambda volume, point:
            features.patch(volume, point, kernel_shape)
    )

    # Create net.
    net = nolearn.lasagne.NeuralNet(
        layers = [

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
    bins, prob_bins = maps.probability_bins(volumes, scale=0.75)

    # Define batch size.
    batch_size = 1000

    # Define a factor to ensure training batches are balanced.
    factor = 0.2

    # Iterate through and train, making sure the batch is balanced.
    for volume in volumes[0:-2]:
        ext.set_volume(volume)
        prob_map = maps.probability_map(volume, bins, prob_bins)
        for input_batch, output_batch, _ in ext.iterate(batch_size,
                                                        point_map=prob_map):
            num_positives = np.count_nonzero(output_batch)
            min_positives = batch_size * factor
            max_positives = batch_size * (1 - factor)
            if num_positives > min_positives and num_positives < max_positives:
                net.fit(input_batch, output_batch)
    print('Finished training.')

    # Test on the second to last volume.
    training_volume = volumes[-2]
    ext.set_volume(training_volume)

    print('Performing segmentation...')
    predicted_volume = ext.predict(net)
    print('Segmentation complete.')

    # Save predicted volume.
    volumetools.pickle_volume(predicted_volume, '2layer9trainnewiter')

    # Compare segmentations.
    training_volume.show_slice(0)
    predicted_volume.show_slice(0)


def third(train=True):

    # List and load all volumes, then switch them to the axial orientation.
    volume_list = volumetools.list_volumes()
    volumes = [volumetools.load_volume(volume) for volume in volume_list]
    for volume in volumes:
        volume.switch_plane('axial')

    # Take a slice corresponding to the location of the left nipple.
    volumes = [volume[int(volume.landmarks['Left nipple'][0])]
               for volume in volumes]

    # Create an Extractor.
    ext = volumetools.Extractor()

    # Add features.
    ext.add_feature(
        feature_name='local_patch',
        feature_function=lambda volume, point:
        features.patch(volume, point, [1, 25, 25])
    )
    ext.add_feature(
        feature_name='context_patch',
        feature_function=lambda volume, point:
        features.scaled_patch(volume, point, [1, 50, 50], [1, 25, 25])
    )
    ext.add_feature(
        feature_name='sternal_angle',
        feature_function=lambda volume, point:
            features.landmark_displacement(volume, point, 'Sternal angle')
    )
    ext.add_feature(
        feature_name='left_nipple',
        feature_function=lambda volume, point:
        features.landmark_displacement(volume, point, 'Left nipple')
    )
    ext.add_feature(
        feature_name='right_nipple',
        feature_function=lambda volume, point:
        features.landmark_displacement(volume, point, 'Right nipple')
    )

    # Create the net.
    net = nolearn.lasagne.NeuralNet(
        layers = [

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
            ext.set_volume(volume)
            for input_batch, output_batch, _ in ext.iterate(
                    batch_size, point_map=maps.half_half_map(volume)):
                net.fit(input_batch, output_batch)
            print('\nFinished training on volume #' + str(index) + '.\n')

        print('Finished training.')

        # Save the parameters for later use.
        net.save_params_to(os.path.join(datapath.get(),
                                        'networks',
                                        'third'))

    else:

        # Load and initialise the net for predictions.
        net.load_params_from(os.path.join(datapath.get(),
                                          'networks',
                                          'third'))
        net.initialize()


    # Test on the reserved second to last volume.
    testing_volume = volumes[-2]
    ext.set_volume(testing_volume)

    # Perform the prediction.
    print('Performing test segmentation.')
    predicted_volume = ext.predict(net, batch_size=batch_size)
    print('Segmentation complete.')

    # Save predicted volume for analysis.
    volumetools.pickle_volume(predicted_volume, 'third')
    testing_volume.show_slice(0)
    predicted_volume.show_slice(0)


def fourth(train=True):

    # List and load all volumes, then switch them to the axial orientation.
    volume_list = volumetools.list_volumes()
    volumes = [volumetools.load_volume(volume) for volume in volume_list]
    for volume in volumes:
        volume.switch_plane('axial')

    # Take a slice corresponding to the location of the left nipple.
    volumes = [volume[int(volume.landmarks['Left nipple'][0])]
               for volume in volumes]

    # Create an Extractor.
    ext = volumetools.Extractor()

    # Create kernel and add features.
    kernel_shape = [1, 21, 21]
    ext.add_feature(
        feature_name='patch',
        feature_function=lambda volume, point:
        features.patch(volume, point, kernel_shape)
    )
    ext.add_feature(
        feature_name='sternal_angle',
        feature_function=lambda volume, point:
            features.landmark_displacement(volume, point, 'Sternal angle')
    )
    ext.add_feature(
        feature_name='left_nipple',
        feature_function=lambda volume, point:
        features.landmark_displacement(volume, point, 'Left nipple')
    )
    ext.add_feature(
        feature_name='right_nipple',
        feature_function=lambda volume, point:
        features.landmark_displacement(volume, point, 'Right nipple')
    )
    ext.add_feature(
        feature_name='back_skin',
        feature_function=lambda volume, point:
        features.landmark_displacement(volume, point, 'Back skin')
    )
    ext.add_feature(
        feature_name='left_humerus_ball',
        feature_function=lambda volume, point:
        features.landmark_displacement(volume, point, 'Left humerus ball')
    )
    ext.add_feature(
        feature_name='right_humerus_ball',
        feature_function=lambda volume, point:
        features.landmark_displacement(volume, point, 'Right humerus ball')
    )
    ext.add_feature(
        feature_name='spinal_cord',
        feature_function=lambda volume, point:
        features.landmark_displacement(volume, point, 'Spinal cord')
    )
    ext.add_feature(
        feature_name='sternal_angle_skin',
        feature_function=lambda volume, point:
        features.landmark_displacement(volume, point, 'Sternal angle skin')
    )
    ext.add_feature(
        feature_name='Sternum superior',
        feature_function=lambda volume, point:
        features.landmark_displacement(volume, point, 'Sternum superior')
    )

    # Create the net.
    net = nolearn.lasagne.NeuralNet(
        layers=[

            # Layers for the patch.
            (lasagne.layers.InputLayer,
             {'name': 'patch', 'shape': tuple([None] + kernel_shape)}),
            (lasagne.layers.Conv2DLayer,
             {'name': 'patch_conv1', 'num_filters': 200,
              'filter_size': (5, 5)}),
            (lasagne.layers.MaxPool2DLayer,
             {'name': 'patch_pool1', 'pool_size': (2, 2)}),
            (lasagne.layers.Conv2DLayer,
             {'name': 'patch_conv2', 'num_filters': 150,
              'filter_size': (3, 3)}),
            (lasagne.layers.MaxPool2DLayer,
             {'name': 'patch_pool2', 'pool_size': (2, 2)}),
            (lasagne.layers.DenseLayer,
             {'name': 'patch_dense1', 'num_units': 150}),

            # Layers for the landmark displacements.
            (lasagne.layers.InputLayer,
             {'name': 'sternal_angle', 'shape': (None, 3)}),
            (lasagne.layers.DenseLayer,
             {'name': 'sternal_angle_dense1', 'num_units': 25}),
            (lasagne.layers.InputLayer,
             {'name': 'left_nipple', 'shape': (None, 3)}),
            (lasagne.layers.DenseLayer,
             {'name': 'left_nipple_dense1', 'num_units': 25}),
            (lasagne.layers.InputLayer,
             {'name': 'right_nipple', 'shape': (None, 3)}),
            (lasagne.layers.DenseLayer,
             {'name': 'right_nipple_dense1', 'num_units': 25}),
            (lasagne.layers.InputLayer,
             {'name': 'back_skin', 'shape': (None, 3)}),
            (lasagne.layers.DenseLayer,
             {'name': 'back_skin_dense1', 'num_units': 25}),
            (lasagne.layers.InputLayer,
             {'name': 'left_humerus_ball', 'shape': (None, 3)}),
            (lasagne.layers.DenseLayer,
             {'name': 'left_humerus_ball_dense1', 'num_units': 25}),
            (lasagne.layers.InputLayer,
             {'name': 'right_humerus_ball', 'shape': (None, 3)}),
            (lasagne.layers.DenseLayer,
             {'name': 'right_humerus_ball_dense1', 'num_units': 25}),
            (lasagne.layers.InputLayer,
             {'name': 'spinal_cord', 'shape': (None, 3)}),
            (lasagne.layers.DenseLayer,
             {'name': 'spinal_cord_dense1', 'num_units': 25}),
            (lasagne.layers.InputLayer,
             {'name': 'sternal_angle_skin', 'shape': (None, 3)}),
            (lasagne.layers.DenseLayer,
             {'name': 'sternal_angle_skin_dense1', 'num_units': 25}),
            (lasagne.layers.InputLayer,
             {'name': 'sternum_superior', 'shape': (None, 3)}),
            (lasagne.layers.DenseLayer,
             {'name': 'sternum_superior_dense1', 'num_units': 25}),

            # Layers for concatenation and output.
            (lasagne.layers.ConcatLayer,
             {'incomings': ['patch_dense1', 'sternal_angle_dense1',
                            'left_nipple_dense1', 'right_nipple_dense1',
                            'back_skin_dense1', 'left_humerus_ball_dense1',
                            'right_humerus_ball_dense1', 'spinal_cord_dense1',
                            'sternal_angle_skin_dense1',
                            'sternum_superior_dense1']}),
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
            print('\nStarting volume #' + str(index) + '\n')
            ext.set_volume(volume)
            for input_batch, output_batch, _ in ext.iterate(
                    batch_size, point_map=maps.half_half_map(volume)):
                net.fit(input_batch, output_batch)

        print('Finished training.')

        # Save the parameters for later use.
        net.save_params_to(os.path.join(datapath.get(),
                                        'networks',
                                        'fourth'))

    else:

        # Load and initialise the net for predictions.
        net.load_params_from(os.path.join(datapath.get(),
                                          'networks',
                                          'fourth'))
        net.initialize()


    # Test on the reserved second to last volume.
    testing_volume = volumes[-2]
    ext.set_volume(testing_volume)

    # Perform the prediction.
    print('Performing test segmentation...')
    predicted_volume = ext.predict(net, batch_size=batch_size)
    print('Segmentation complete.')

    # Save predicted volume for analysis.
    volumetools.pickle_volume(predicted_volume, 'fourth')
    testing_volume.show_slice(0)
    predicted_volume.show_slice(0)


if __name__ == '__main__':

    third()