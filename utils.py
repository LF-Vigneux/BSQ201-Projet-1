import pennylane as qml
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from typing import Tuple
from numpy.typing import NDArray


def get_feature_vectors_and_labels(
    dataset_name: str,
    path: str = "",
) -> Tuple[NDArray[np.float_], NDArray[np.float_]]:
    """
    Reads the datasets from a file and divides the data into two matrices. The first one is the
    feature vectors and the othe rone is the labels. The labels of the dataset to read must be
    in the last column.

    Parameters:
    - dataset_name (str): The name of the datasdet file including the extension.
    - path (str): The relative path to the dataset file.

    Returns:
    Tuple[NDArray[np.float_], NDArray[np.float_]]:  - The matrix of the feature vectors
                                                    - The vector of the set's labels
    """
    dataset = np.load(path + dataset_name, allow_pickle=True)
    return dataset[:, :-1], dataset[:, -1]
