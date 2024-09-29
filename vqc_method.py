import numpy as np
from numpy.typing import NDArray
from typing import Tuple
import pennylane as qml
from utils import get_qnode_instance, get_score
from error_functions import mean_square_error


class VQC_Solver:
    def __init__(
        self,
        embedding_circuit: callable,
        ansatz: callable,
        num_params: int,
        num_qubits: int,
    ) -> None:
        self.embedding = embedding_circuit
        self.ansatz = ansatz
        self.params = np.zeros(num_params)
        self.num_qubits = num_qubits

        self.circuit_to_optimize = get_qnode_instance(
            self.create_vqc_circuit, self.num_qubits
        )

    def create_vqc_circuit(
        self, feature_vector: NDArray[np.float_], params: NDArray[np.float_]
    ):
        self.embedding(feature_vector)
        self.ansatz(params)
        return qml.probs(wires=range(self.num_qubits))

    def classification_function(probs_array: NDArray[np.float_]):
        if probs_array[0] < 0.5:
            return 1
        return 0

    def run(
        self,
        feature_vectors: NDArray[np.float_],
        labels: NDArray[np.float_],
        optimizer_function: callable,
        classification_function: callable = classification_function,
        error_function: callable = mean_square_error,
        training_ratio: float = 0.8,
    ):
        training_period = int(training_ratio * len(labels))

        training_vectors = feature_vectors[:training_period, :]
        testing_vectors = feature_vectors[training_period:, :]
        training_labels = labels[:training_period]
        testing_labels = labels[training_period:]

        predictions = np.empty_like(testing_labels)

        # Optimising the ansatz
        def cost_function(
            params: NDArray[np.float_],
        ):
            resulting_labels = np.empty_like(training_labels)
            for i, training_vector in enumerate(training_vectors):
                probs = self.circuit_to_optimize(training_vector, params)
                resulting_labels[i] = classification_function(probs)
            return error_function(resulting_labels, training_labels)

        self.params = optimizer_function(cost_function, self.params).x

        # Getting the predictions
        for i, testing_vector in enumerate(testing_vectors):
            predictions[i] = classification_function(
                self.circuit_to_optimize(testing_vector, self.params)
            )

        return get_score(predictions, testing_labels), predictions
