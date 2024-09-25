import pennylane as qml
from pennylane import QNode
import numpy as np
from typing import Tuple
from numpy.typing import NDArray

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
    return dataset[:, :-1], dataset[:, -1]


def transform_vector_into_power_of_two_dim(a: NDArray[np.float_]):
    if not np.log2(len(a)) % 1 == 0:
        power_of_two = int(np.ceil(np.log2(len(a))))
        new_dim = 2**power_of_two
        new_a = np.zeros(new_dim)
        new_a[: len(a)] = a
        return new_a
    return a


# Essayer peut etre avec lambda de specifier le nombre de qubits
def get_qnode_instance(
    circuit_function: callable,
    num_qubits: int,
) -> QNode:
    dev = qml.device("default.qubit", wires=num_qubits)
    return qml.QNode(circuit_function, dev)


def get_score(prediction_labels: NDArray[np.float_], true_lables: NDArray[np.float_]):
    score = 0
    for pred, true_value in zip(prediction_labels, true_lables):
        if pred == true_value:
            score += 1
    score /= len(prediction_labels)
    return score


def get_accuracies(
    predicted_labels: NDArray[np.float_], expirement_labels: NDArray[np.float_]
) -> Tuple[int, int, int, int]:
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0

    for pred, true_value in zip(predicted_labels, expirement_labels):
        if pred == 0:
            if true_value == 0:
                true_negative += 1
            else:
                false_negative += 1
        else:
            if true_value == 1:
                true_positive += 1
            else:
                false_positive += 1
    return true_positive, false_positive, true_negative, false_negative


def get_good_distribution_of_labels(
    feature_vectors: NDArray[np.float_], labels: NDArray[np.float_], number_per_label
) -> Tuple[NDArray[np.float_], NDArray[np.float_]]:
    label_zero_indixes = np.where(labels == 0)[
        0
    ]  # 4 premières lignes chat m'a aidé? ok?
    label_one_indixes = np.where(labels == 1)[0]

    selected_label_zero = np.random.choice(
        label_zero_indixes, number_per_label, replace=False
    )
    selected_label_one = np.random.choice(
        label_one_indixes, number_per_label, replace=False
    )
    indexes_list = np.append(selected_label_zero, selected_label_one)
    np.random.shuffle(indexes_list)

    return (feature_vectors[indexes_list, :], labels[indexes_list])
