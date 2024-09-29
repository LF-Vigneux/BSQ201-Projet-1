import pennylane as qml
import numpy as np
from numpy.typing import NDArray
from utils import transform_vector_into_power_of_two_dim


def angle_embedding(a: NDArray[np.float_], num_qubits: int = None, rotation: str = "Y"):
    for i, theta in enumerate(a):
        if rotation == "Y":
            qml.RY(theta, wires=(i % num_qubits))
        elif rotation == "X":
            qml.RX(theta, wires=(i % num_qubits))
        else:
            qml.RZ(theta, wires=(i % num_qubits))


def amplitude_embedding(a: NDArray[np.float_]):
    new_a = transform_vector_into_power_of_two_dim(a)
    num_qubits = int(np.log2(len(new_a)))

    qml.AmplitudeEmbedding(features=new_a, wires=range(num_qubits), normalize=True)


def iqp_embedding(a: NDArray[np.float_]):
    qubits = range(len(a))
    qml.IQPEmbedding(a, wires=qubits)
