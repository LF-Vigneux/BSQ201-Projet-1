import pennylane as qml
from pennylane import QNode
import numpy as np
from typing import Tuple
from numpy.typing import NDArray
from scipy.special import softmax
import csv


def get_feature_vectors_and_labels(
    dataset_name: str, extension: str = "npy", path: str = "", rows_to_skip: int = 0
) -> Tuple[NDArray[np.float_], NDArray[np.float_]]:
    """
    Reads the datasets from a file and divides the data into two matrices. The first one is the
    feature vectors and the othe rone is the labels. The labels of the dataset to read must be
    in the last column.

    Parameters:
    - dataset_name (str): The name of the datasdet file without the extension.
    - extension (str="npy"): The extension of the file of the dataset
    - path (str=""): The relative path to the dataset file.

    Returns:
    Tuple[NDArray[np.float_], NDArray[np.float_]]:  - The matrix of the feature vectors
                                                    - The vector of the set's labels
    """
    if extension == "csv":
        dataset = np.loadtxt(
            path + dataset_name + "." + extension, delimiter=",", skiprows=rows_to_skip
        )
    else:
        dataset = np.load(path + dataset_name + "." + extension, allow_pickle=True)

    # Soft max mais les données entre 0 et pi
    return softmax(dataset[:, :-1]) * np.pi, dataset[:, -1]


def transform_vector_into_power_of_two_dim(a):
    if not np.log2(len(a)) % 1 == 0:
        power_of_two = int(np.ceil(np.log2(len(a))))
        new_dim = 2**power_of_two
        new_a = np.zeros(new_dim)
        new_a[: len(a)] = a
        return new_a
    return a


# Essayer peut etre avec lambda de specifier le nombre de qubits
def get_qnode_instance(
    embedding_circuit: callable,
    num_qubits: int,
) -> QNode:
    dev = qml.device("default.qubit", wires=num_qubits)
    return qml.QNode(embedding_circuit, dev)


# Utiliser d'autres tests comme dans mon stage réservoir??? PAS MASE par contre
def mean_square_error(predicted_label, expirement_labels):
    total = 0
    for true_value, predicted in zip(predicted_label, expirement_labels):
        total += (
            predicted - true_value
        ) ** 2  # Juste des réels donc pas de norme right?
    return total / (2 * len(predicted_label))
