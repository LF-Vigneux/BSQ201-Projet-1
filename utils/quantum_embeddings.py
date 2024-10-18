"""
File containing some quantum embeddings. The function creates their Pennylane implementation.
To run this algorithms, a Qnode needs to be created.
"""

import pennylane as qml
import numpy as np
from numpy.typing import NDArray
from .utils import transform_vector_into_power_of_two_dim


def angle_embedding(
    a: NDArray[np.float_], num_qubits: int = None, rotation: str = "Y"
) -> None:
    """
    Circuit of an angle embedding of a feature vector for a given number of qubits and a given rotation axis. The features are assigned
    to the ith % num_qubits qubit.
     - a (NDArray[np.float_]): The feature vector to encode.
     - num_qubits (int): The number of qubits of the encoding
     - rotation (str = "Y"): The rotation axis of the angle gates.
     Returns:
     None
    """
    for i, theta in enumerate(a):
        if rotation == "Y":
            qml.RY(theta, wires=(i % num_qubits))
        elif rotation == "X":
            qml.RX(theta, wires=(i % num_qubits))
        else:
            qml.RZ(theta, wires=(i % num_qubits))


def amplitude_embedding(a: NDArray[np.float_]):
    """
    Circuit of an amplitude embedding of a feature vector.
     - a (NDArray[np.float_]): The feature vector to encode.
     Returns:
     None
    """
    new_a = transform_vector_into_power_of_two_dim(a)
    num_qubits = int(np.log2(len(new_a)))

    qml.AmplitudeEmbedding(features=new_a, wires=range(num_qubits), normalize=True)


def iqp_embedding(a: NDArray[np.float_]):
    """
    Circuit of an IQP embedding of a feature vector.
     - a (NDArray[np.float_]): The feature vector to encode.
     Returns:
     None
    """
    qubits = range(len(a))
    qml.IQPEmbedding(a, wires=qubits)
