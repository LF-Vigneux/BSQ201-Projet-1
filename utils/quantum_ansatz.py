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
