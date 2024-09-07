import pennylane as qml
from pennylane import QNode
import numpy as np
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
    return dataset[:, :-1], np.matrix.flatten(dataset[:, -1])


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
