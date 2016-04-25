import os

from . import general


def pickle_network(net, filename):
    full_filename = os.path.join(general.datapath(), 'networks', filename)
    general.pickle_object(net, full_filename)


def unpickle_network(filename):
    full_filename = os.path.join(general.datapath(), 'networks', filename)
    return general.unpickle_object(full_filename)
