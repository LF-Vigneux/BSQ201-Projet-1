import numpy as np
import pandas as pd
from utils.utils import (
    get_score,
    get_feature_vectors_and_labels,
    normalize_feature_vectors,
    get_good_distribution_of_labels,
)
from utils.quantum_ansatz import ansatz_circuit
from utils.error_functions import precision, negative_prediction_value, informedness
from utils.quantum_embeddings import angle_embedding, iqp_embedding, amplitude_embedding
from scipy.optimize import minimize
from utils.error_functions import mean_square_error
from pennylane.optimize import (
    NesterovMomentumOptimizer,
    AdamOptimizer,
    GradientDescentOptimizer,
)
from kernel_method import Quantum_Kernel_Classification
from vqc_method import VQC_Solver
from qcnn_method import QCNN_Solver


# Load the dataset
feature_vectors_pulsar, labels_pulsar = get_feature_vectors_and_labels(
    "HTRU_2", extension="csv", path="datasets/", rows_to_skip=0
)

# Load the dataset
feature_vectors_tele, labels_tele = get_feature_vectors_and_labels(
    "magic_gamma_telescope", extension="csv", path="datasets/", rows_to_skip=1
)

# Change the 0 labels into -1
labels_pulsar = (2 * labels_pulsar) - 1
labels_tele = np.where(labels_tele == "g", 1, -1)


def minimisation_cobyla(cost_function, params):
    result = minimize(
        cost_function,
        params,
        method="COBYLA",
        options={
            "maxiter": 70,
            "disp": True,  # To reduce the time of optimization
        },
    )
    return result.x


def minimisation_nes(cost_function, params):
    result = NesterovMomentumOptimizer(stepsize=2)
    new_params = params
    for _ in range(100):
        new_params = result.step(cost_function, new_params)
    return new_params


def tester(
    feature_vectors,
    labels,
    embedding,
    num_qubits,
    method,
    optimizer_function=minimisation_nes,
    training_ratio=0.8,
    ansatz=ansatz_circuit,
    num_params=0,
):
    # Get a good sample of the dataset since it is to big
    feature_vectors, labels = get_good_distribution_of_labels(
        feature_vectors, labels, 100
    )
    # Normalize the feature vectors
    feature_vectors = normalize_feature_vectors(feature_vectors)
    training_period = int(len(labels) * training_ratio)
    expirement_labels = labels[training_period:]

    return_array = np.empty(4)

    if method == "QSVM":
        classifier = Quantum_Kernel_Classification(embedding, num_qubits)
        return_array[0], predictions = classifier.run(
            feature_vectors, labels, training_ratio
        )
    elif method == "VQC":
        classifier = VQC_Solver(embedding, ansatz, num_params)
        return_array[0], predictions = classifier.run(
            feature_vectors, labels, optimizer_function, training_ratio=training_ratio
        )
    else:
        classifier = QCNN_Solver(embedding, num_qubits)
        return_array[0], predictions = classifier.run(
            feature_vectors, labels, optimizer_function, training_ratio=training_ratio
        )

    return_array[1] = precision(predictions, expirement_labels)
    return_array[2] = negative_prediction_value(predictions, expirement_labels)
    return_array[3] = informedness(predictions, expirement_labels)

    return return_array


num_tests = 10
test_return = np.empty((num_tests, 4))
i = 0

"""
Pulsars
"""

num_qubits = 8


def angle_embedding_x(a):
    return angle_embedding(a, num_qubits=num_qubits, rotation="X")


def angle_embedding_y(a):
    return angle_embedding(a, num_qubits=num_qubits, rotation="Y")


def angle_embedding_z(a):
    return angle_embedding(a, num_qubits=num_qubits, rotation="Z")


# Itération 1
test_return[i, :] = tester(
    feature_vectors_pulsar,
    labels_pulsar,
    amplitude_embedding,
    3,
    "QSVM",
)
print("iteration ", i + 1, "terminée")
i += 1

# Itération 2
test_return[i, :] = tester(
    feature_vectors_pulsar,
    labels_pulsar,
    angle_embedding_x,
    8,
    "QSVM",
)
print("iteration ", i + 1, "terminée")
i += 1

# Itération 3
test_return[i, :] = tester(
    feature_vectors_pulsar,
    labels_pulsar,
    angle_embedding_y,
    8,
    "QSVM",
)
print("iteration ", i + 1, "terminée")
i += 1

# Itération 4
test_return[i, :] = tester(
    feature_vectors_pulsar,
    labels_pulsar,
    angle_embedding_z,
    8,
    "QSVM",
)
print("iteration ", i + 1, "terminée")
i += 1

# Itération 5
test_return[i, :] = tester(
    feature_vectors_pulsar,
    labels_pulsar,
    iqp_embedding,
    8,
    "QSVM",
)
print("iteration ", i + 1, "terminée")
i += 1


"""
Telescope
"""

num_qubits = 10


def angle_embedding_x(a):
    return angle_embedding(a, num_qubits=num_qubits, rotation="X")


def angle_embedding_y(a):
    return angle_embedding(a, num_qubits=num_qubits, rotation="Y")


def angle_embedding_z(a):
    return angle_embedding(a, num_qubits=num_qubits, rotation="Z")


# Itération 1
test_return[i, :] = tester(
    feature_vectors_tele,
    labels_tele,
    amplitude_embedding,
    4,
    "QSVM",
)
print("iteration ", i + 1, "terminée")
i += 1

# Itération 2
test_return[i, :] = tester(
    feature_vectors_tele,
    labels_tele,
    angle_embedding_x,
    10,
    "QSVM",
)
print("iteration ", i + 1, "terminée")
i += 1

# Itération 3
test_return[i, :] = tester(
    feature_vectors_tele,
    labels_tele,
    angle_embedding_y,
    10,
    "QSVM",
)
print("iteration ", i + 1, "terminée")
i += 1

# Itération 4
test_return[i, :] = tester(
    feature_vectors_tele,
    labels_tele,
    angle_embedding_z,
    10,
    "QSVM",
)
print("iteration ", i + 1, "terminée")
i += 1

# Itération 5
test_return[i, :] = tester(
    feature_vectors_tele,
    labels_tele,
    iqp_embedding,
    10,
    "QSVM",
)
print("iteration ", i + 1, "terminée")
i += 1


# Convert the array to a pandas DataFrame
df = pd.DataFrame(test_return)

# Save the DataFrame to a CSV file
df.to_csv("run_results.csv", index=False)
