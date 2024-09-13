import numpy as np
from numpy.typing import NDArray
from kernel_method import get_kernel_prediction, get_kernel_embedding
from vqc_method import get_vqc_result
import quantum_embeddings
from utils import get_feature_vectors_and_labels, get_qnode_instance

import pennylane as qml

from pennylane.templates import RandomLayers  # QCNN

# Package à télécharger... Tout les optimiseurs sans gradients de Powell
import pdfo

from scipy.optimize import minimize, OptimizeResult, Bounds

# À voir si SPSA ou PSO utile, utiliser optimizer.py de mon stage cet été.


def main(
    feature_vectors: NDArray[np.float_], labels: NDArray[np.int_], training_period: int
):

    num_qubits = 4

    # To use the kernel_angle_embedding function correctly, you need to use a wrapper functions with the number of qubits and the rotation gate to use
    rotation = "Y"

    def angle_embedding(a):
        return quantum_embeddings.angle_embedding(
            a, num_qubits=num_qubits, rotation=rotation
        )

    # The other embeddings in the quantum_kernel_embeddings file do not need those wrapper functions. You only need to chose the correct e,mbedding in the line below.
    ########EN CLASSES ÇA VA ËTRE PLUS CLEAN

    def kernel_embedding(a, b):
        return get_kernel_embedding(a, b, angle_embedding, num_qubits)
        # return get_kernel_embedding(a,b,quantum_embeddings.amplitude_embedding,num_qubits)
        # return get_kernel_embedding(a,b,quantum_embeddings.iqp_embedding,num_qubits)

    # À voir s'il serait mieux de faire une classe et juste de l'appeler, aurait directv accès au qnode. PT mélanger avec le embedding, on le choisi et on fait juste plug le calable dedans.
    # Comme ça qnode instance appelé dans la classe... à voir PT même classes au lieu de fonctions d'embedding. Comme ça direct accès nombre de qubits, pas nombre cohérent avec QNode

    circuit = get_qnode_instance(kernel_embedding, num_qubits)

    def qkernel(A, B):
        return np.array([[circuit(a, b)[0] for b in B] for a in A])

    score, predictions = get_kernel_prediction(
        qkernel, feature_vectors, labels, training_period
    )

    print("The score of the kernel: ", score)
    print("The predictions of the labels: ", predictions)
    print("The true value of the labels: ", labels[training_period:])


if __name__ == "__main__":
    feature_vectors, labels = get_feature_vectors_and_labels(
        "HTRU_2", extension="csv", path="datasets/", rows_to_skip=0
    )

    # Réduire dataset, trop gros:
    feature_vectors = feature_vectors[1962:2462, :]
    labels = labels[1962:2462]

    training_period = int(len(labels) * 0.8)
    main(feature_vectors, labels, training_period)
