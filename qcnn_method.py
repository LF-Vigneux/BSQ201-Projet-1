import numpy as np
from numpy.typing import NDArray
from typing import Tuple
import pennylane as qml
from utils import get_qnode_instance, mean_square_error, get_score


class QCNN_Solver:
    def __init__(
        self,
        embedding_circuit: callable,
        convolution_circuit: callable,
        num_qubits: int,
    ) -> None:
        self.embedding = embedding_circuit
        self.num_qubits = num_qubits
        self.convolution_circuit = convolution_circuit
        self.circuit_to_optimize = get_qnode_instance(
            self.generate_qcnn_circuit, self.num_qubits
        )
        self.params = np.zeros(
            self.num_params_utilized
        )  # Marche pour le moment où pas de param de pool

    # General pooling function that works for even or odd number of bits, always filling with last qubit conneted with the first on the Cnot
    # If odd number of qubits, median qubit-> no operation
    def pool(self, old_size: int):
        new_size = np.ceil(old_size / 2)
        for i in range(new_size, old_size):
            qml.CNOT(wires=(i, old_size - i - 1))
        return new_size

    def generate_qcnn_circuit(self, feature_vector, params):
        self.num_params_utilized = 0  # petit test inutile
        num_qubits_utilized = self.num_qubits
        self.embedding(feature_vector)
        while True:
            self.num_params_utilized += self.convolution_circuit(
                params[self.num_params_utilized :]
            )  # Comment savoir le nombre de paramètres, approche, donne tout et retourne le nombre d'utilisé...
            num_qubits_utilized = self.pool(
                num_qubits_utilized
            )  # jsp si ça va marcher ça
            if num_qubits_utilized == 1:
                break
        return qml.probs(wires=0)

    def classification_function(probs_array):
        if probs_array[0] > 0.5:
            return 1
        return 0

    def run(
        self,
        feature_vectors: NDArray[np.float_],
        labels: NDArray[np.float_],
        optimizer_function: callable,
        classification_function: callable = classification_function,
        batched_data: Tuple[bool, int] = (  # Faire le cas pas batched
            True,
            10,
        ),  # If to batch the data and if yes, how many batched data used
        training_ratio: float = 0.8,
    ):
        training_period = int(training_ratio * len(labels))

        training_vectors = feature_vectors[:training_period, :]
        testing_vectors = feature_vectors[training_period:, :]
        training_labels = labels[:training_period]
        testing_labels = labels[training_period:]

        predictions = np.empty_like(testing_labels)

        batch_lenght = int(len(training_labels) / batched_data[1])

        # Optimising the weights of the interactions
        batch_number = 0

        def cost_function(
            params,
        ):  # À voir si on la sort de la classe et juste la donner
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
            return mean_square_error(
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
