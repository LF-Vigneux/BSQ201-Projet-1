import numpy as np
import pennylane as qml
from sklearn.svm import SVC
from numpy.typing import NDArray
from utils import get_qnode_instance


# Mettres les fonctions dans run
class Quantum_Kernel_Classification:
    def __init__(self, embedding_function: callable, num_qubits) -> None:
        # opérations pour créer circuit total
        # Trouver comment avoir le nombre de qubit du circuit
        self.circuit = get_qnode_instance(embedding_function, num_qubits)


def get_kernel_prediction(
    qkernel: callable,
    feature_vectors: NDArray[np.float_],
    labels: NDArray[np.int_],
    training_period: int,
):
    training_vectors = feature_vectors[:training_period, :]
    testing_vecors = feature_vectors[training_period:, :]
    training_labels = labels[:training_period]
    testing_labels = labels[training_period:]

    model = SVC(kernel=qkernel)
    model.fit(training_vectors, training_labels)

    score = model.score(testing_vecors, testing_labels)
    predictions = model.predict(testing_vecors)

    return score, predictions


def get_kernel_embedding(a, b, embedding_circuit: callable, num_qubits: int):
    embedding_circuit(a)
    qml.adjoint(embedding_circuit(b))
    return qml.probs(wires=num_qubits)
