import theano
import theano.tensor as T
import lasagne
import datatools as dt
import volumetools as vt
import features as ft
import networks as nt
import numpy as np
import scipy.misc


def first():
    
    data_path = dt.read_data_path()
    volume_list = dt.list_volumes(data_path)
    volumes = [dt.load_volume(volume) for volume in volume_list]

    prob_map = vt.build_prob_map(volumes)

    it = vt.DataProcessor(volumes[0])

    kernel_shape = [3, 3, 3]
    it.add_feature(
        lambda volume, point: ft.patch(volume, point, kernel_shape)
    )
    it.add_feature(
        lambda volume, point: ft.intensity_mean(volume, point, kernel_shape)
    )
    it.add_feature(
        lambda volume, point: ft.probability(volume, point, prob_map)
    )

    map = np.array(prob_map, dtype='bool')
    batch_size = 100
    # x = it.get_training_data(batch_size, map)
    # y = next(x)

    # Create Theano variables for input and target minibatch.
    input_var = T.matrix('X', dtype='float64')
    target_var = T.vector('y', dtype='int64')

    network = nt.basic((batch_size, it.get_vector_size()), input_var)

    # Create loss function.
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean() + \
           1e-4 * lasagne.regularization.regularize_network_params(
               network, lasagne.regularization.l2)

    # Create parameter update expressions.
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(loss, params,
                                                learning_rate=0.01,
                                                momentum=0.9)

    # Compile training function to update parameters and returns training loss.
    train_fn = theano.function([input_var, target_var], loss, updates=updates)

    # Train network.
    training_data = it.iterate(batch_size, map)
    for epoch in range(1):
        loss = 0
        length = 0
        for input_batch, target_batch, _ in training_data:
            loss += train_fn(input_batch, target_batch)
            length += 1
            if length % 100 == 0:
                print(length, 'batches done')
                
        print("Epoch %d: Loss %g" % (epoch + 1, loss / length))

    # Use trained network for predictions.
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    predict = theano.function([input_var], T.argmax(test_prediction, axis=1))

    it_test = vt.DataProcessor(volumes[0][190:200])
    it_test.add_feature(
        lambda volume, point: ft.patch(volume, point, kernel_shape)
    )
    it_test.add_feature(
        lambda volume, point: ft.intensity_mean(volume, point, kernel_shape)
    )
    it_test.add_feature(
        lambda volume, point: ft.probability(volume, point, prob_map)
    )

    test_data = it_test.get_test_volume(predict)
    scipy.misc.toimage(test_data[0]).show()

if __name__ == '__main__':

    first()