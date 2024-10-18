"""
File containing two quantum ansatz to use in the VQC algorithm. The first one is inspired by the 2-local ansatz.
IBM Quantum. TwoLocal. url : https://docs.quantum.ibm.com/api/qiskit/qiskit.
circuit.library.TwoLocal#twolocal (Consulted 17/10/2024).

The second is a ansatz based on Pennylane's RandomLayers.
"""

import pennylane as qml
import numpy as np
from numpy.typing import NDArray


def ansatz_circuit(params: NDArray[np.float_]) -> int:
    """
    8 qubit ansatz that works well with the HTRU_2 dataset. It is implemented with the Pennylane architecture.
    Parameters:
    - params (NDArray[np.float_]): The value of the parametrized gates of the ansatz.
    Returns:
    int: The number of parameters used in the ansatz.
    """

    params_no = 0
    num_qubits = 3

    for j in range(num_qubits):
        qml.Hadamard(j)

    for i in range(num_qubits):
        qml.RX(params[params_no], i)
        params_no += 1

    for k in range(num_qubits - 1):
        qml.CNOT(wires=[k, k + 1])

    qml.CNOT(wires=[num_qubits - 1, 0])

    for x in range(num_qubits):
        qml.RX(params[params_no], x)
        params_no += 1

    for y in range(num_qubits):
        qml.RY(params[params_no], y)
        params_no += 1

    for z in range(num_qubits):
        qml.RZ(params[params_no], z)
        params_no += 1
    return params_no


def ansatz_random_layer(
    params: NDArray[np.float_], num_qubits: int = 3, num_params_per_qubits: int = 4
) -> None:
    """
    Creates a random ansatz with the random layer of Pennylane.
    Parameters:
    - params (NDArray[np.float_]): The value of the parametrized gates of the ansatz.
    - num_qubits (int = 3): The number of qubits of the ansatz.
    - num_params_per_qubits (int = 4): The number of parametrized gates per qubit.

    Returns:
    None
    """

    params_reshaped = params.reshape(num_qubits, num_params_per_qubits)

    qml.RandomLayers(weights=params_reshaped, wires=range(2), ratio_imprim=0.75)
