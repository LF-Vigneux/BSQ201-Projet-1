"""
File containing miscellaneous functions that can be used by the user and the classifiers.
There are, for example, some functions to load a dataset and normalize it, create qnodes and get 
the exactitude of the classifiers.
"""

import pennylane as qml
from pennylane import QNode
import numpy as np
from typing import Tuple
from numpy.typing import NDArray
import pandas as pd


def get_feature_vectors_and_labels(
    dataset_name: str, extension: str = "npy", path: str = "", rows_to_skip: int = 0
) -> Tuple[NDArray[np.float_], NDArray[np.float_]]:
    """
    Reads the datasets from a csv or npy file and divides the data into two arrays. The first one is the
    feature vectors, and the other one is the labels. The labels of the dataset to read must be
    in the last column.

    Parameters:
    - dataset_name (str): The name of the dataset file without the extension.
    - extension (str="npy"): The extension of the file of the dataset.
    - path (str=""): The relative path to the dataset file.

    Returns:
    Tuple[NDArray[np.float_], NDArray[np.float_]]:  - The matrix of the feature vectors.
                                                    - The vector of the set's labels.
    """
    if extension == "csv":
        dataset = pd.read_csv(path + dataset_name + "." + extension)
        feature_vectors = dataset.iloc[:, :-1].to_numpy()
        labels = dataset.iloc[:, -1].to_numpy()
    else:
        dataset = np.load(path + dataset_name + "." + extension, allow_pickle=True)
        feature_vectors = dataset[rows_to_skip:, :-1]
        labels = dataset[rows_to_skip:, -1]

    return feature_vectors, labels


def transform_vector_into_power_of_two_dim(
    feature_vector: NDArray[np.float_],
) -> NDArray[np.float_]:
    """
    Transform a feature vector into one with the same information but to the superior or equal power of two dim.
    The remaining elements of the new array are filled with zeroes.

    Parameters:
    - feature_vector: NDArray[np.float_]: The feature vector to put into a power of two dimension vector.

    Returns:
    NDArray[np.float_]: The new feature vector into a power of two arrays.
    """
    if not np.log2(len(feature_vector)) % 1 == 0:
        power_of_two = int(np.ceil(np.log2(len(feature_vector))))
        new_dim = 2**power_of_two
        new_a = np.zeros(new_dim)
        new_a[: len(feature_vector)] = feature_vector
        return new_a
    return feature_vector


def get_qnode_instance(
    circuit_function: callable, num_qubits: int, device_name: str = "default.qubit"
) -> QNode:
    """
    Transforms the Python function describing a Pennylane circuit into a qnode with the specified device.
    Parameters:
    - circuit_function (callable): The Python function describing the Pennylane circuit
    - num_qubits (int): The number of qubits of the circuit
    - device_name (str="default.qubit"): The name of the device being that will run the circuit. It must be valid with the qml.device function.
    Returns:
    QNode: The quantum node of the circuit that can now be run.
    """
    dev = qml.device(device_name, wires=num_qubits)
    return qml.QNode(circuit_function, dev)


def get_score(
    prediction_labels: NDArray[np.float_], true_lables: NDArray[np.float_]
) -> int:
    """
    Gets the number of accurately predicted labels by the prediction.
    Parameters:
    - prediction_labels (NDArray[np.float_]): The labels predicted by the machine learning classifier.
    - true_lables: NDArray[np.float_]: The expected labels (The theoretical results).
    Returns:
    int: The number of correctly predicted labels.
    """
    score = 0
    for pred, true_value in zip(prediction_labels, true_lables):
        if pred == true_value:
            score += 1
    score /= len(prediction_labels)
    return score


def get_accuracies(
    predicted_labels: NDArray[np.float_], expirement_labels: NDArray[np.float_]
) -> Tuple[int, int, int, int]:
    """
    Defines the number of false positives, true positives, false negatives and true negatives of the prediction labels.
    Parameters:
    - prediction_labels (NDArray[np.float_]): The labels predicted by the machine learning classifier.
    - true_lables: NDArray[np.float_]: The expected labels (The theoretical results).
    Returns:
    Tuple[int, int, int, int]:  - The number of correctly predicted 1 results (True positive).
                                - The number of incorrectly predicted 1 results (False positive).
                                - The number of correctly predicted -1 results (True negative).
                                - The number of incorrectly predicted -1 results (False negative).
    """
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0

    for pred, true_value in zip(predicted_labels, expirement_labels):
        if pred == -1:
            if true_value == -1:
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
    feature_vectors: NDArray[np.float_],
    labels: NDArray[np.float_],
    number_per_label: int,
) -> Tuple[NDArray[np.float_], NDArray[np.float_]]:
    """
    Gets a distribution of the feature vector given with the same number as label "1" feature vector than "-1".
    The choice of the vectors is random. The final array is then shuffled so that the -1 and 1  vectors are randomly ordered.

    Parameters:
    - feature_vectors (NDArray[np.float_]): The feature vectors of the dataset.
    - labels (NDArray[np.float_]): The associated labels of this dataset (in the same order as the feature vectors).
    - number_per_label (int): The number of feature vectors to get per label.

    Returns:
    Tuple[NDArray[np.float_], NDArray[np.float_]]:  - The matrix of the balanced feature vectors.
                                                    - The vector of the balanced set's labels.
    """
    label_negatives_indixes = np.where(labels == -1)[0]
    label_positives_indixes = np.where(labels == 1)[0]

    selected_label_negative = np.random.choice(
        label_negatives_indixes, number_per_label, replace=False
    )
    selected_label_positive = np.random.choice(
        label_positives_indixes, number_per_label, replace=False
    )
    indexes_list = np.append(selected_label_negative, selected_label_positive)
    np.random.shuffle(indexes_list)

    return (feature_vectors[indexes_list, :], labels[indexes_list])


def normalize_feature_vectors(
    feature_vectors: NDArray[np.float_],
) -> NDArray[np.float_]:
    """
    Normalizes the feature vector so that each feature has a minimum value of -1 and a maximum value of 1.

    Parameters:
    - feature_vectors (NDArray[np.float_]): The feature vectors of the dataset that need to be normalized.
    Returns:
    NDArray[np.float_]:  The normalized feature vectors.
    """
    normalized_feature_vectors = np.empty_like(feature_vectors)
    for i in range(np.shape(feature_vectors)[1]):
        normalized_feature_vectors[:, i] = feature_vectors[:, i] / np.linalg.norm(
            feature_vectors[:, i]
        )
    return normalized_feature_vectors
