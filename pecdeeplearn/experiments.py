import theano
import theano.tensor as T
import lasagne
import datatools as dt
import volumetools as vt
import features as ft
import networks as nt


def first():
    
    data_path = dt.read_data_path()
    volume = dt.load_volume(dt.list_volumes(data_path)[0])
    volume.switch_plane('axial')
    train_vol = volume[250:300]
    test_vol = volume[300]

    train_vol.show_slice(0)

    it = vt.BatchIterator(volume)

    it.add_feature(
        lambda volume, point: ft.patch(volume, point, [])
    )

    # create Theano variables for input and target minibatch
    input_var = T.tensor4('X', dtype='int32')
    target_var = T.vector('y', dtype='int32')

    network = networks.basic(input_shape, input_var)

    # create loss function
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean() + 1e-4 * lasagne.regularization.regularize_network_params(
            network, lasagne.regularization.l2)

    # create parameter update expressions
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=0.01,
                                                momentum=0.9)

    # compile training function that updates parameters and returns training loss
    train_fn = theano.function([input_var, target_var], loss, updates=updates)

    # train network (assuming you've got some training data in numpy arrays)
    training_data = small_vol.iterate_batches(kernel_shape, 100)
    for epoch in range(1):
        loss = 0
        length = 0
        for input_batch, target_batch in training_data:
            loss += train_fn(input_batch, target_batch)
            length += 1
            if length % 100 == 0:
                print(length, 'batches done')
                
        print("Epoch %d: Loss %g" % (epoch + 1, loss / length))

    # use trained network for predictions
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    predict_fn = theano.function([input_var], T.argmax(test_prediction, axis=1))
    ##print("Predicted class for first test input: %r" % predict_fn(test_data[0]))

