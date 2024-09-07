import pennylane as qml
import numpy as np
from utils import transform_vector_into_power_of_two_dim


def angle_embedding(
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
    new_a = transform_vector_into_power_of_two_dim(a)
    new_b = transform_vector_into_power_of_two_dim(b)
    num_qubits = int(np.log2(len(new_a)))
    qubits = range(num_qubits)

    qml.AmplitudeEmbedding(new_a, wires=qubits, normalize=True)
    qml.adjoint(qml.AmplitudeEmbedding(new_b, wires=qubits, normalize=True))
    return qml.probs(wires=qubits)


def iqp_embedding(a, b):
    qubits = range(len(a))
    qml.IQPEmbedding(a, wires=qubits)
    qml.adjoint(qml.IQPEmbedding(b, wires=qubits))
    return qml.probs(wires=qubits)
