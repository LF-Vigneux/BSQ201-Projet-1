import numpy as np
from numpy.typing import NDArray
from kernel_method import Quantum_Kernel_Classification
from vqc_method import VQC_Solver
from qcnn_method import QCNN_Solver
import quantum_embeddings
import quantum_ansatz
from utils import get_feature_vectors_and_labels
from pennylane.templates import RandomLayers  # QCNN
import pennylane as qml

# Package à télécharger... Tout les optimiseurs sans gradients de Powell
import pdfo

from scipy.optimize import minimize, OptimizeResult, Bounds

# À voir si SPSA ou PSO utile, utiliser optimizer.py de mon stage cet été.


def main(
    feature_vectors: NDArray[np.float_], labels: NDArray[np.int_], training_ratio: int
):
    num_qubits = 8
    """KERNELS"""

    # To use the kernel_angle_embedding function correctly, you need to use a wrapper functions with the number of qubits and the rotation gate to use
    rotation = "Y"

    def angle_embedding(a):
        return quantum_embeddings.angle_embedding(
            a, num_qubits=num_qubits, rotation=rotation
        )

    """
    The class is then called, the number of qubits need to be coherent with the embedding:
    angle_embedding: no qubits restrictions
    amplitude_embedding: qubits muste be the base two log of the input. This number must be rounded up to the next integer
    iqp_embedding: The number of qubits must be the same as the number of features

    The IQP and angle must be normalized in 0 to pi() and the others juste be normalized
    """
    # feature_vectors = feature_vectors * np.pi
    # kernel_qml = Quantum_Kernel_Classification(angle_embedding, num_qubits)
    # kernel_qml = Quantum_Kernel_Classification(
    #   quantum_embeddings.iqp_embedding, num_qubits
    # )

    kernel_qml = Quantum_Kernel_Classification(
        quantum_embeddings.amplitude_embedding, num_qubits
    )

    score, predictions = kernel_qml.run(
        feature_vectors, labels, training_ratio=training_ratio
    )

    training_period = int(len(labels) * training_ratio)

    print("The score of the kernel: ", score)
    print("The predictions of the labels: ", predictions)
    print("The true value of the labels: ", labels[training_period:])

    """""" """""" """""" """""" ""
    """VQC"""
    """
    num_params_ansatz = "À remplir"
    vqc = VQC_Solver(
        quantum_embeddings.iqp_embedding,
        quantum_ansatz.test,
        num_params_ansatz,
        num_qubits,
    )

    # The minimiser needs to have only the cost function and params as parameters
    def minimisation(cost_function, params):
        return minimize(cost_function, params, method="COBYLA")

    score, predictions = vqc.run(
        feature_vectors, labels, minimisation, training_ratio=training_ratio
    )

    print("The score of the VQC: ", score)
    print("The predictions of the labels: ", predictions)
    print("The true value of the labels: ", labels[training_period:])
    """

    """""" """""" """""" """"""
    """QCNN"""
    batches = 10
    
    dev = qml.device('default.qubit', wires=num_qubits)
    rand_params = np.random.uniform(high=2 * np.pi, size=(num_qubits))
    qml.qnode(dev)
    def convolution_circuit(params):
        for i in range(num_qubits):
            qml.RY(params[i], wires=i)
        
        
        for j in range (int(num_qubits/2.5)):
            qml.CNOT(wires=[j, j + 5])
            qml.RY(params[j], wires=j+5)

        for k in range (int(num_qubits/2.5), num_qubits):
            qml.RY(params[k], wires= k - 3)
            qml.CNOT(wires= [k, k-3])
        

        return convolution_circuit

    qcnn = QCNN_Solver(
        quantum_embeddings.iqp_embedding, convolution_circuit, num_qubits
    )

    # The minimiser needs to have only the cost function and params as parameters
    def minimisation(cost_function, params):
        return minimize(
            cost_function, params, method="COBYLA", options={"maxiter": batches}
        )  # Jsp si maxiter va vraiment limiter le nombre d'évaluations de la fonction de coût, sinon utiliser un gradient descent plus facile

    score, predictions = qcnn.run(
        feature_vectors,
        labels,
        minimisation,
        batched_data=(True, batches),
        training_ratio=training_ratio,
    )

    print("The score of the QCNN: ", score)
    print("The predictions of the labels: ", predictions)
    print("The true value of the labels: ", labels[training_period:])


if __name__ == "__main__":
    feature_vectors, labels = get_feature_vectors_and_labels(
        "HTRU_2", extension="csv", path="datasets/", rows_to_skip=0
    )

    # Réduire dataset, trop gros:
    feature_vectors = feature_vectors[2000:2100, :]
    labels = labels[2000:2100]
    # normalize feature vectors
    feature_vectors = np.array(
        [
            feature_vectors[i, :] / np.linalg.norm(feature_vectors[i, :])
            for i in range(np.shape(feature_vectors)[0])
        ]
    )

    print(np.where(labels == 1))

    training_ratio = 0.8
    main(feature_vectors, labels, training_ratio)
