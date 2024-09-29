import numpy as np
from numpy.typing import NDArray
from typing import Tuple
import pennylane as qml
from utils import get_qnode_instance, get_score
from error_functions import mean_square_error


class QCNN_Solver:
    def __init__(
        self,
        embedding_circuit: callable,
        num_qubits: int,
    ) -> None:
        self.embedding = embedding_circuit
        self.num_qubits = num_qubits
        self.circuit_to_optimize = get_qnode_instance(
            self.generate_qcnn_circuit, self.num_qubits
        )
        self.num_qubits

        # get number of parameters
        self.num_params = 0
        test_qubits = self.num_qubits
        while test_qubits > 1:
            self.num_params += 2 * test_qubits
            test_qubits = int(np.ceil(test_qubits / 2))
        self.num_params = self.num_params - 2
        self.params = np.zeros(self.num_params)

        # Marche pour le moment où pas de param de pool

    # General pooling function that works for even or odd number of bits, always filling with last qubit conneted with the first on the Cnot
    # If odd number of qubits, median qubit-> no operation
    def pool(self, old_size: int):
        new_size = int(np.ceil(old_size / 2))
        for i in range(new_size, old_size):
            qml.CNOT(wires=(i, old_size - i - 1))
        return new_size

    def convolution(self, num_qubits, params):
        if num_qubits == 2:
            qml.RY(params[0], 0)
            qml.RY(params[0], 1)
            qml.CNOT((0, 1))
            return 2
        else:
            qubit_to_target = max(num_qubits - 3, 1)
            num_param_per_qubit = np.ones(num_qubits)
            for j in range(num_qubits):
                qml.RY(params[j], j)
            i_param = num_qubits
            for i in range(num_qubits):
                target_qubit = (i + qubit_to_target) % num_qubits
                qml.CNOT((i, target_qubit))
                if num_param_per_qubit[i] == 1:
                    qml.RY(params[i_param], i)
                    i_param += 1
                    num_param_per_qubit[i] = 2
                if num_param_per_qubit[target_qubit] == 1:
                    qml.RY(params[i_param], target_qubit)
                    i_param += 1
                    num_param_per_qubit[target_qubit] = 2
        return 2 * num_qubits

    def generate_qcnn_circuit(
        self, feature_vector: NDArray[np.float_], params: NDArray[np.float_]
    ):
        num_params_utilized = 0
        num_qubits_utilized = self.num_qubits
        self.embedding(feature_vector)
        while True:
            num_params_utilized += self.convolution(
                num_qubits_utilized,
                params[
                    num_params_utilized:
                ],  # On fait juste enlever ceux utilisés à date
            )
            # qml.Barrier(only_visual=True)   #Pour des tests
            num_qubits_utilized = self.pool(num_qubits_utilized)
            # qml.Barrier(only_visual=True)   #Pour des tests
            if num_qubits_utilized == 1:
                break
        return qml.probs(wires=0)

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

        def cost_function(
            params,
        ):  # À voir si on la sort de la classe et juste la donner

            resulting_labels = np.empty_like(training_labels)
            for i, training_vector in enumerate(training_vectors):
                probs = self.circuit_to_optimize(training_vector, params)
                resulting_labels[i] = classification_function(probs)
            return error_function(
                resulting_labels, training_labels
            )  # C'est comme la fonction de coût à voir si on la mets en param

        self.params = optimizer_function(
            cost_function, self.params
        ).x  # The optimizer needs to stop at the batch number specefied earlier

        # Getting the predictions
        for i, testing_vector in enumerate(testing_vectors):
            predictions[i] = classification_function(
                self.circuit_to_optimize(testing_vector, self.params)
            )

        return get_score(predictions, testing_labels), predictions

    def run_batched(
        self,
        feature_vectors: NDArray[np.float_],
        labels: NDArray[np.float_],
        optimizer_function: callable,
        classification_function: callable = classification_function,
        error_function: callable = mean_square_error,
        num_batches: int = 10,
        training_ratio: float = 0.8,
    ):
        training_period = int(training_ratio * len(labels))

        training_vectors = feature_vectors[:training_period, :]
        testing_vectors = feature_vectors[training_period:, :]
        training_labels = labels[:training_period]
        testing_labels = labels[training_period:]

        predictions = np.empty_like(testing_labels)

        batch_lenght = int(len(training_labels) / num_batches)

        # Optimising the weights of the interactions
        batch_number = 0

        def cost_function(
            params,
        ):  # À voir si on la sort de la classe et juste la donner
            nonlocal batch_number
            batched_training_vectors = training_vectors[
                batch_number * batch_lenght : (batch_number + 1) * batch_lenght,
                :,  # On les mets de la grandeur actuel, dernier pas nécessairement même taille
            ]
            batched_training_labels = training_labels[
                batch_number * batch_lenght : (batch_number + 1) * batch_lenght
            ]

            resulting_labels = np.empty_like(batched_training_labels)
            for i, training_vector in enumerate(batched_training_vectors):
                probs = self.circuit_to_optimize(training_vector, params)
                resulting_labels[i] = classification_function(probs)

            batch_number = batch_number + 1
            return error_function(
                resulting_labels, training_labels
            )  # C'est comme la fonction de coût à voir si on la mets en param

        self.params = optimizer_function(
            cost_function, self.params
        ).x  # The optimizer needs to stop at the batch number specefied earlier

        # Getting the predictions
        for i, testing_vector in enumerate(testing_vectors):
            predictions[i] = classification_function(
                self.circuit_to_optimize(testing_vector, self.params)
            )

        return get_score(predictions, testing_labels), predictions
