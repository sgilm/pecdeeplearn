import pickle
import os.path


def datapath():
    """Read and return the datapath specified in the package root directory."""

    # Get the path to the file containing the datapath.
    full_filename = \
        os.path.join(os.path.dirname(__file__), '..', 'datapath.txt')

    # Open and read the .txt file.
    with open(full_filename, 'r') as f:
        path = f.read().strip()

    return path


def pickle_object(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f, -1)


def unpickle_object(filename):
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
    return obj
