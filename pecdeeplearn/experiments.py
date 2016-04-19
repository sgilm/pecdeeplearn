import lasagne
import volumetools
import features
import numpy as np
import maps
import nolearn.lasagne


def first(train=True):

    volume_list = volumetools.list_volumes()
    volumes = [volumetools.load_volume(volume) for volume in volume_list]
    for volume in volumes:
        volume.switch_plane('axial')

    ext = volumetools.Extractor()

    kernel_shape = [1, 13, 13]
    ext.add_feature(
        feature_name='patch',
        feature_function=lambda volume, point:
            features.patch(volume, point, kernel_shape)
    )
    # ext.add_feature(
    #     feature_name='intensity_mean',
    #     feature_function=lambda volume, point:
    #         features.intensity_mean(volume, point, kernel_shape)
    # )
    # ext.add_feature(
    #     feature_name='sternal_angle',
    #     feature_function=lambda volume, point:
    #         features.landmark_displacement(volume, point, 'sternal_angle')
    # )

    map = maps.segmentation_map(volumes)
    batch_size = 100

    net = NeuralNet(
        layers=[  # three layers: one hidden layer
            ('input', layers.InputLayer),
            ('hidden', layers.DenseLayer),
            ('output', layers.DenseLayer),
        ],
        # layer parameters:
        input_shape=(None, 169),  # 96x96 input pixels per batch
        hidden_num_units=100,  # number of units in hidden layer
        output_nonlinearity=sigmoid,
        output_num_units=1,

        # optimization method:
        update=nesterov_momentum,
        update_learning_rate=0.001,
        update_momentum=0.9,

        regression=False,
        # flag to indicate we're dealing with regression problem
        max_epochs=2,  # we want to train this many epochs
        verbose=1,
    )

    # # Create Theano variables for input and target minibatch.
    # patch_var = T.tensor3('patch', dtype='float64')
    # intensity_var = T.matrix('intensity', dtype='float64')
    # sternal_var = T.matrix('sternal', dtype='float64')
    # target_var = T.vector('y', dtype='int64')

    # # Create loss function.
    # prediction = lasagne.layers.get_output(network)
    # loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    # loss = loss.mean() + \
    #        1e-4 * lasagne.regularization.regularize_network_params(
    #            network, lasagne.regularization.l2)
    #
    # # Create parameter update expressions.
    # params = lasagne.layers.get_all_params(network, trainable=True)
    # updates = lasagne.updates.nesterov_momentum(loss, params,
    #                                             learning_rate=0.01,
    #                                             momentum=0.9)
    #
    # # Compile training function to update parameters and return training loss.
    # train_fn = theano.function([input_var, target_var], loss, updates=updates)

    for volume in volumes[0:1]:
        ext.set_volume(volume)
        for input_batch, output_batch, _ in ext.iterate(batch_size, point_map=map):
            # output_batch = output_batch.reshape((batch_size, 1))
            X = input_batch['patch'].reshape(100, 169).astype(np.float32)
            output_batch = output_batch.astype(np.int32)
            net.fit(X, output_batch)

    weights = lasagne.layers.get_all_param_values(network)
    np.save('weights', weights)

    # Use trained network for predictions.
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    predict = theano.function([input_var], T.argmax(test_prediction, axis=1))

    ext.volume = volumes[-1]

    test_volume = ext.prediction(predict, map)

    for i in range(10):
        scipy.misc.toimage(ext.volume.seg_data[150 + i]).show()
        scipy.misc.toimage(test_volume[150 + i]).show()


def second():

    volume_list = volumetools.list_volumes()
    volumes = [volumetools.load_volume(volume) for volume in volume_list]
    for volume in volumes:
        volume.switch_plane('axial')

    volumes = [volume[121] for volume in volumes]

    ext = volumetools.Extractor()

    kernel_shape = [1, 21, 21]
    ext.add_feature(
        feature_name='patch',
        feature_function=lambda volume, point:
            features.patch(volume, point, kernel_shape)
    )

    batch_size = 1000

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
        verbose=True
    )

    bins, prob_bins = maps.probability_bins(volumes, scale=0.75)
    factor = 0.2

    for volume in volumes[0:-1]:
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

    training_volume = volumes[-1]
    ext.set_volume(training_volume)

    print('Performing segmentation...')
    predicted_volume = ext.predict(net)

    print('Segmentation complete.')

    volumetools.pickle_volume(predicted_volume, '2layer9trainnewiter')

    training_volume.show_slice(0)
    predicted_volume.show_slice(0)


def third():

    # List and load all volumes, then switch them to the axial orientation.
    volume_list = volumetools.list_volumes()
    volumes = [volumetools.load_volume(volume) for volume in volume_list]
    for volume in volumes:
        volume.switch_plane('axial')

    # Take a slice corresponding to the location of the left nipple.
    volumes = [volume[int(volume.landmarks['Left nipple'][0])] for volume in volumes]

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

    # Create the net.
    net = nolearn.lasagne.NeuralNet(
        layers = [

            # Layers for the patch.
            (lasagne.layers.InputLayer, {'name': 'patch', 'shape': tuple([None] + kernel_shape)}),
            (lasagne.layers.Conv2DLayer, {'name': 'patch_conv1', 'num_filters': 100, 'filter_size': (5, 5)}),
            (lasagne.layers.MaxPool2DLayer, {'name': 'patch_pool1', 'pool_size': (2, 2)}),
            (lasagne.layers.Conv2DLayer, {'name': 'patch_conv2', 'num_filters': 100, 'filter_size': (3, 3)}),
            (lasagne.layers.MaxPool2DLayer, {'name': 'patch_pool2', 'pool_size': (2, 2)}),
            (lasagne.layers.DenseLayer, {'name': 'patch_dense1', 'num_units': 100}),

            # Layers for the landmark displacements.
            (lasagne.layers.InputLayer, {'name': 'sternal_angle', 'shape': (None, 3)}),
            (lasagne.layers.DenseLayer, {'name': 'sternal_angle_dense1', 'num_units': 50}),
            (lasagne.layers.InputLayer, {'name': 'left_nipple', 'shape': (None, 3)}),
            (lasagne.layers.DenseLayer, {'name': 'left_nipple_dense1', 'num_units': 50}),
            (lasagne.layers.InputLayer, {'name': 'right_nipple', 'shape': (None, 3)}),
            (lasagne.layers.DenseLayer, {'name': 'right_nipple_dense1', 'num_units': 50}),

            # Layers for concatenation and output.
            (lasagne.layers.ConcatLayer, {'incomings': ['patch_dense1', 'sternal_angle_dense1', 'left_nipple_dense1', 'right_nipple_dense1']}),
            (lasagne.layers.DenseLayer, {'name': 'output', 'num_units': 2, 'nonlinearity': lasagne.nonlinearities.softmax}),

        ],

        # Define learning parameters.
        update=lasagne.updates.nesterov_momentum,
        update_learning_rate=0.0001,
        update_momentum=0.9,

        # Define training parameters.
        max_epochs=100,
        verbose=True
    )

    # Define the batch size.
    batch_size = 100

    # Train on all but the last volume, and use a half-half map.
    for volume in volumes[0:-1]:
        ext.set_volume(volume)
        for input_batch, output_batch, _ \
                in ext.iterate(batch_size, point_map=maps.half_half_map(volume)):
            net.fit(input_batch, output_batch)

    print('Finished training.')

    # Test on the reserved final volume.
    testing_volume = volumes[-1]
    ext.set_volume(testing_volume)

    # Perform the prediction.
    print('Performing test segmentation...')
    predicted_volume = ext.predict(net)
    print('Segmentation complete.')

    # Save predicted volume for analysis.
    volumetools.pickle_volume(predicted_volume, 'test')
    testing_volume.show_slice(0)
    predicted_volume.show_slice(0)


if __name__ == '__main__':

    third()