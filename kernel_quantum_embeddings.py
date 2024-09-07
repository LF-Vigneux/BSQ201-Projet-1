import pennylane as qml
from pennylane import QNode
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from typing import Tuple
from numpy.typing import NDArray
from utils import get_feature_vectors_and_labels


# Essayer peut etre avec lambda de specifier le nombre de qubits
def get_qnode_instance(
    embedding_circuit: callable,
    num_qubits: int,
) -> QNode:
    dev = qml.device("default.qubit", wires=num_qubits)
    return qml.QNode(embedding_circuit, dev)


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
            qml.adjoint(qml.RY(theta, wires=i % num_qubits))
        elif rotation == "X":
            qml.adjoint(qml.RX(theta, wires=i % num_qubits))
        else:
            qml.adjoint(qml.RZ(theta, wires=i % num_qubits))

    return qml.probs(wires=range(num_qubits))


def amplitude_embedding(a, b):
    num_qubits = np.ceil(np.log2(len(a)))
    qubits = range(int(num_qubits))
    qml.AmplitudeEmbedding(a, wires=qubits)
    qml.adjoint(qml.AmplitudeEmbedding(b), wires=qubits)
    return qml.probs(wires=qubits)


def iqb_embedding(a, b):
    qubits = range(len(a))
    qml.IQPEmbedding(a, wires=qubits)
    qml.adjoint(qml.IQPEmbedding(b), wires=qubits)
    return qml.probs(wires=qubits)


def qkernel(A, B):
    return np.array([[get_qnode_instance(a, b)[0] for b in B] for a in A])


test = get_qnode_instance(amplitude_embedding, 2)
print(test((0, 0, 0, 1), (1, 0, 0, 0)))
