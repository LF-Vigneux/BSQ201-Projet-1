import pennylane as qml
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from typing import Tuple
from numpy.typing import NDArray
from utils import get_feature_vectors_and_labels


def kernel_matrix(
    feature_vectors: NDArray[np.float_],
    embedding_circuit: callable,
    num_qubits: int = None,
):
    if num_qubits is None:
        num_qubits = np.shape(feature_vectors)[0]

    dev = qml.device("default.qubit", wires=num_qubits)
    circuit = qml.QNode(embedding_circuit, dev)

    return


# Comment bien faire le node, j'essaie de généraliser mais pas sur comment, prblèàme de decorator


def kernel_angle_embedding(
    a, b, num_qubits: int = None, rotation: str = "Y"
):  # Amplitude, juste pas game de tout changer le nom
    for i, theta in enumerate(a):
        if rotation == "Y":
            qml.RY(theta, wires=i % num_qubits)
        elif rotation == "X":
            qml.RX(theta, wires=i % num_qubits)
        else:
            qml.RZ(theta, wires=i % num_qubits)
    for i, theta in reversed(enumerate(b)):
        if rotation == "Y":
            qml.RY(theta, wires=i % num_qubits)
        elif rotation == "X":
            qml.RX(theta, wires=i % num_qubits)
        else:
            qml.RZ(theta, wires=i % num_qubits)

    return qml.probs(wires=range(num_qubits))


def qkernel(A, B):
    return np.array([[kernel_angle_embedding(a, b)[0] for b in B] for a in A])
