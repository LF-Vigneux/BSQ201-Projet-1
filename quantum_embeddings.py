import pennylane as qml
import numpy as np
from utils import transform_vector_into_power_of_two_dim


def angle_embedding(a, num_qubits: int = None, rotation: str = "Y"):
    for i, theta in enumerate(a):
        if rotation == "Y":
            qml.RY(theta, wires=i % num_qubits)
        elif rotation == "X":
            qml.RX(theta, wires=i % num_qubits)
        else:
            qml.RZ(theta, wires=i % num_qubits)


def amplitude_embedding(a, b):
    new_a = transform_vector_into_power_of_two_dim(a)
    num_qubits = int(np.log2(len(new_a)))
    qubits = range(num_qubits)

    qml.AmplitudeEmbedding(new_a, wires=qubits, normalize=True)


def iqp_embedding(a):
    qubits = range(len(a))
    qml.IQPEmbedding(a, wires=qubits)
