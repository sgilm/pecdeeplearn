import pickle


def datapath():
    """Read and return the datapath specified in the root directory."""

    # Open and read the .txt file.
    with open('../../datapath.txt', 'r') as f:
        path = f.read().strip()

    return path


def pickle_object(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f, -1)


def unpickle_object(filename):
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
    return obj
