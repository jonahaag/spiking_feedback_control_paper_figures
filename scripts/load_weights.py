"""Load a per-batch weight .dat file as a numpy array.

Usage: python load_weights.py path/to/weights_in_hid1_fine.dat
The array shape is read from trainlog.pkl in the same folder.
"""
import os, pickle, sys
import numpy as np

path = sys.argv[1]
name = os.path.splitext(os.path.basename(path))[0]
meta = pickle.load(open(os.path.join(os.path.dirname(path), "trainlog.pkl"), "rb"))
shape = tuple(meta["arrays"][name])

weights = np.fromfile(path, dtype="float64").reshape(shape)
print(name, weights.shape)
