import numpy as np
import pennylane as qml
from sklearn.svm import SVC
from numpy.typing import NDArray
from utils import get_qnode_instance


# Mettres les fonctions dans run
class Quantum_Kernel_Classification:
    def __init__(self, embedding_circuit: callable, num_qubits: int) -> None:
        self.embedding = embedding_circuit
        self.num_qubits = num_qubits
        self.kernel_circuit = get_qnode_instance(
            self.get_kernel_embedding, self.num_qubits
        )

    def run(
        self,
        feature_vectors: NDArray[np.float_],
        labels: NDArray[np.float_],
        training_ratio: float = 0.8,
        svm=SVC,
    ):
        def qkernel(A, B):
            return np.array([[self.kernel_circuit(a, b)[0] for b in B] for a in A])

        training_period = int(training_ratio * len(labels))

        training_vectors = feature_vectors[:training_period, :]
        testing_vecors = feature_vectors[training_period:, :]
        training_labels = labels[:training_period]
        testing_labels = labels[training_period:]

        model = svm(kernel=qkernel)
        model.fit(training_vectors, training_labels)

        score = model.score(testing_vecors, testing_labels)
        predictions = model.predict(testing_vecors)

        return score, predictions

    def get_kernel_embedding(self, a, b):
        self.embedding(a)
        qml.adjoint(self.embedding)(b)
        return qml.probs(wires=range(self.num_qubits))
