import lasagne
import theano.tensor as T


def basic(input_shape, input_var=None):

    network = lasagne.layers.InputLayer(input_shape, input_var)
    network = lasagne.layers.DenseLayer(network, num_units=200)
    network = lasagne.layers.DenseLayer(network, num_units=2,nonlinearity=T.nnet.softmax)

    return network
