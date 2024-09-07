import numpy as np
from numpy.typing import NDArray
from kernel_method import get_kernel_prediction
import quantum_kernel_embeddings
from utils import get_feature_vectors_and_labels, get_qnode_instance


def main(
    feature_vectors: NDArray[np.float_], labels: NDArray[np.int_], training_period: int
):

    num_qubits = 8

    # To use the kernel_angle_embedding function correctly, you need to use a wrapper function with the number of qubits and the rotation gate to use
    rotation = "Y"

    def angle_embedding(a, b):
        return quantum_kernel_embeddings.angle_embedding(
            a, b, num_qubits=num_qubits, rotation=rotation
        )

    circuit = get_qnode_instance(angle_embedding, num_qubits)

    # The other embeddings in the quantum_kernel_embeddings file do not need those wrapper functions. Only one of the next two lines is needed.
    # The number of qubits specified need to be coherent with the embedding stategy employed. See the embeddings functions descriptions for more details.

    # circuit = get_qnode_instance(quantum_kernel_embeddings.amplitude_embedding, num_qubits)
    # circuit = get_qnode_instance(quantum_kernel_embeddings.iqp_embedding, num_qubits)

    # À voir s'il serait mieux de fair eune classe et juste de l'appeler, aurait directv accès au qnode. PT mélanger avec le embedding, on le choisi et on fait juste plug le calable dedans.
    # Comme ça qnode instance appelé dans la classe... à voir PT même classes au lieu de fonctions d'embedding. Comme ça direct accès nombre de qubits, pas nombre cohérent avec QNode

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
        "pulsar_stars", extension="csv", path="datasets/"
    )

    # Réduire dataset, trop gros:
    feature_vectors = feature_vectors[1962:2062, :]
    labels = labels[1962:2062]

    training_period = int(len(labels) * 0.8)
    main(feature_vectors, labels, training_period)
