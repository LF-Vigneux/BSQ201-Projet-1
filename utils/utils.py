import pennylane as qml
from pennylane import QNode
import numpy as np
from typing import Tuple
from numpy.typing import NDArray


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


def transform_vector_into_power_of_two_dim(
    feature_vector: NDArray[np.float_],
) -> NDArray[np.float_]:
    """
    Transform a feature vector into one with the same information but to the the superior ou equal power of two dim.
    The remaining elements of the new array are filled with zeroes.

    Parameters:
    - feature_vector: NDArray[np.float_]: The feature vector to put into a power of two dimension vector.

    Returns:
    NDArray[np.float_]: The new feature vector into a power of two array.
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
    Transforms the a python function describing a pennylane circuit into a qnode with the specified device.
    Parameters:
    - circuit_function (callable): The python function descibing the pennylane circuit
    - num_qubits (int): The number of qubits of the circuit
    - device_name (str="default.qubit"): The name of the device being that will run the circuit. It must be valid with the qml.device function.
    Returns:
    QNode: The quantum node of the circuit ready to be runned.
    """
    dev = qml.device(device_name, wires=num_qubits)
    return qml.QNode(circuit_function, dev)


def get_score(
    prediction_labels: NDArray[np.float_], true_lables: NDArray[np.float_]
) -> int:
    """
    Gets the number of accuratly predicted labls by the prediction.
    Parameters:
    - prediction_labels (NDArray[np.float_]): The labels predicted by the machine learning classifier.
    - true_lables: NDArray[np.float_]: The expected labels (The theoritical results).
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
    Definines each predicted reult as false or true positive or negative result
    Parameters:
    - prediction_labels (NDArray[np.float_]): The labels predicted by the machine learning classifier.
    - true_lables: NDArray[np.float_]: The expected labels (The theoritical results).
    Returns:
    Tuple[int, int, int, int]:  - The number of correctly predicted 1 results (True positive)
                                - The number of incorrectly predicted 1 results (False positive)
                                - The number of correctly predicted 0 results (True negative)
                                - The number of incorrectly predicted 0 results (False negative)
    """
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
    feature_vectors: NDArray[np.float_],
    labels: NDArray[np.float_],
    number_per_label: int,
) -> Tuple[NDArray[np.float_], NDArray[np.float_]]:
    """
    Gets a distribution of the feature vector given with the same nomber as label "1" feature vector than "0".
    The choice of the vectors are random. The final array is then shuffled so that the 0 and 1 are randomly placed.

    Parameters:
    - feature_vectors (NDArray[np.float_]): The feature vectors of the dataset.
    - labels (NDArray[np.float_]): The associated labels of this dataset (in the same order as the feature vectors).
    - number_per_label (int): The number of feature vector to get per label.

    Returns:
    Tuple[NDArray[np.float_], NDArray[np.float_]]:  - The matrix of the balanced feature vectors
                                                    - The vector of the balenced set's labels
    """
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


def normalize_feature_vectors(
    feature_vectors: NDArray[np.float_],
) -> NDArray[np.float_]:
    """
    Normalizes the feature vector so that each feature as a minimum value of -1 and maximum value of 1

    Parameters:
    - feature_vectors (NDArray[np.float_]): The feature vectors of the dataset that need to be normalized.
    Returns:
    NDArray[np.float_]:  The normalized feature vectors
    """
    normalized_feature_vectors = np.empty_like(feature_vectors)
    for i in range(np.shape(feature_vectors)[1]):
        normalized_feature_vectors[:, i] = feature_vectors[:, i] / np.linalg.norm(
            feature_vectors[:, i]
        )
    return normalized_feature_vectors
