import numpy as np
import pennylane as qml
from numpy.typing import NDArray
from kernel_method import Quantum_Kernel_Classification
from vqc_method import VQC_Solver
from qcnn_method import QCNN_Solver
import utils.quantum_embeddings
import utils.quantum_ansatz
from utils.utils import get_feature_vectors_and_labels, get_good_distribution_of_labels
from pennylane.templates import RandomLayers  # QCNN

# Package à télécharger... Tout les optimiseurs sans gradients de Powell
import pdfo

from scipy.optimize import minimize, OptimizeResult, Bounds

# À voir si SPSA ou PSO utile, utiliser optimizer.py de mon stage cet été.


def main(
    feature_vectors: NDArray[np.float_], labels: NDArray[np.int_], training_ratio: int
):
    num_qubits = 8
    """KERNELS"""
    print("Running QSVM")

    # To use the kernel_angle_embedding function correctly, you need to use a wrapper functions with the number of qubits and the rotation gate to use
    rotation = "Y"

    def angle_embedding(a):
        return utils.quantum_embeddings.angle_embedding(
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
        utils.quantum_embeddings.amplitude_embedding, num_qubits
    )

    score, predictions = kernel_qml.run(
        feature_vectors, labels, training_ratio=training_ratio
    )

    training_period = int(len(labels) * training_ratio)

    print("The score of the kernel: ", score)
    print("The predictions of the labels: ", predictions)
    print("The true value of the labels: ", labels[training_period:])
    print()

    """""" """""" """""" """""" ""
    print("Running VQC")
    """VQC"""
    num_params = 12
    vqc = VQC_Solver(
        utils.quantum_embeddings.amplitude_embedding,
        utils.quantum_ansatz.ansatz_circuit,
        num_params,
        num_qubits,
    )

    # The minimiser needs to have only the cost function and params as parameters
    def minimisation(cost_function, params):
        return minimize(
            cost_function,
            params,
            method="COBYLA",
            options={"tol": 1e-08},
        )

    score, predictions = vqc.run(
        feature_vectors, labels, minimisation, training_ratio=training_ratio
    )

    print("The score of the VQC: ", score)
    print("The predictions of the labels: ", predictions)
    print("The true value of the labels: ", labels[training_period:])
    print()

    """""" """""" """""" """"""
    """QCNN"""
    print("QCNN is running")
    batches = 15

    qcnn = QCNN_Solver(utils.quantum_embeddings.iqp_embedding, num_qubits)

    # The minimiser needs to have only the cost function and params as parameters
    def minimisation(cost_function, params):
        return minimize(
            cost_function,
            params,
            method="COBYLA",
            options={
                # "maxiter": batches,     #À enlever si on veut batch
                "maxiter": 30,
            },
        )  # Jsp si maxiter va vraiment limiter le nombre d'évaluations de la fonction de coût, sinon utiliser un gradient descent plus facile

    score, predictions = qcnn.run(
        feature_vectors,
        labels,
        minimisation,
        # batched_data=batches,
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
    feature_vectors, labels = get_good_distribution_of_labels(
        feature_vectors, labels, 50
    )
    # normalize feature vectors
    for i in range(np.shape(feature_vectors)[1]):
        feature_vectors[:, i] = feature_vectors[:, i] / np.linalg.norm(
            feature_vectors[:, i]
        )

    training_ratio = 0.8
    main(feature_vectors, labels, training_ratio)
