"""
File containing the class of the variationnal quantum method. The class of this classifier and its method are all in this file.
The function that runs the main algorithm is the .run method.
"""

from pennylane import numpy as np
from numpy.typing import NDArray
from typing import Tuple, List
import pennylane as qml
from utils.utils import get_qnode_instance, get_score
from utils.error_functions import mean_square_error


class VQC_Solver:
    def __init__(
        self,
        embedding_circuit: callable,
        ansatz: callable,
        num_params: int,
        num_qubits: int,
    ) -> None:
        """
        Object that can run the quantum variational classification algorithm.

        Parameters:
        - embedding_circuit (callable): The Python function describing the embedding circuit of the data. It must use the Pennylane architecture to create the circuit.
        - ansatz (callable): The Python function describing the ansatz to be optimized by the algorithm.
        - num_params (int): The number of paramters only in the ansatz function.
        - num_qubits (int): The number of qubits of the embedding circuit and the ansatz.

        Returns:
        None
        """
        self.embedding = embedding_circuit
        self.ansatz = ansatz
        self.params = np.zeros(num_params)
        self.num_qubits = num_qubits

        self.circuit_to_optimize = get_qnode_instance(
            self.create_vqc_circuit, self.num_qubits
        )

    def create_vqc_circuit(
        self, feature_vector: NDArray[np.float_], params: NDArray[np.float_]
    ) -> List[float]:
        """
        Method that creates the VQC circuit to be used by the class.

        Parameters:
        - self: The VQC_Solver object that will use this circuit.
        - feature_vector (NDArray[np.float_]): The feature vector to be encoded in the instance of the circuit.
        - params (NDArray[np.float_]): The parameters to be assigned to each parametrized gate of the ansatz.

        Returns:
        List[float]: The probabilities associated with each basis state in the circuit. They will not be directly accessible
                     since a QNode needs to be created with this function to work.
        """
        self.embedding(feature_vector)
        self.ansatz(params)
        z_pauli = qml.PauliZ(0)
        for i in range(1, self.num_qubits):
            z_pauli = z_pauli @ qml.PauliZ(i)
        return qml.expval(z_pauli)

    def run(
        self,
        feature_vectors: NDArray[np.float_],
        labels: NDArray[np.float_],
        optimizer_function: callable,
        error_function: callable = mean_square_error,
        training_ratio: float = 0.8,
    ) -> Tuple[int, NDArray[np.int_]]:
        """
        Method to run the variationnal quatum classifier algorithm. By using a training dataset, for a set of training vectors,
        it will predict their associated labels.

        Parameters:
        - self: The VQC_Solver object to call the method on.
        - feature_vectors (NDArray[np.float_]):  The feature vectors used to train the classifier. The prediction vectors are also in this array. They are after the training ones.
        - labels: (NDArray[np.float_]): The labels associated with the feature vectors. The ones given for the prediction phase will be used
                                        to determine the precision of the classifier. The labels must be in the same order as their associated feature vector in the feature_vectors matrix.
                                        The value of each label must be -1 or 1.
        - optimizer_function (callable): The function that optimizes the cost function with a given set of parameters. It must have only two parameters in this order:
                                         the cost function to optimize and the parameter array to be used. The function must return the optimized parameters.
        - error_function (callable = mean_square_error): The function that takes for input the labels given by the classifier and their real value and gives a numeric value of exactness. This function is then optimized.
                                                         The optimizer will tweak the parameters to minimize the result of that function. The function must use directly the expectation values in the calculation. In
                                                         other words, it can not transform the prediction data to calculate the cost.
        - training_ratio (float = 0.8): The ratio between the number of feature vectors used for training on the total number of feature vectors.

        Returns:
        Tuple[int, NDArray[np.int_]]:  - The number of correctly predicted labels
                                       - The prediction labels of the testing feature vectors.
        """
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
            resulting_labels = [
                self.circuit_to_optimize(training_vector, params)
                for training_vector in training_vectors
            ]
            return error_function(resulting_labels, training_labels)

        self.params = optimizer_function(cost_function, self.params)

        # Getting the predictions
        for i, testing_vector in enumerate(testing_vectors):
            predictions[i] = np.sign(
                self.circuit_to_optimize(testing_vector, self.params)
            )

        return get_score(predictions, testing_labels), predictions
