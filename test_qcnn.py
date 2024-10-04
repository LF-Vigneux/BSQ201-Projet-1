import numpy as np
from utils.utils import (
    get_feature_vectors_and_labels,
    get_good_distribution_of_labels,
    normalize_feature_vectors,
)

# Load the dataset
feature_vectors, labels = get_feature_vectors_and_labels(
    "HTRU_2", extension="csv", path="datasets/", rows_to_skip=0
)

# Get a good sample of the dataset since it is to big
feature_vectors, labels = get_good_distribution_of_labels(feature_vectors, labels, 50)
# Normalize the feature vectors
feature_vectors = normalize_feature_vectors(feature_vectors)


from qcnn_method import QCNN_Solver
import utils.quantum_embeddings

num_qubits = 8


def angle(feature_vectors):
    return utils.quantum_embeddings.angle_embedding(feature_vectors, 8, rotation="Y")


qcnn = QCNN_Solver(utils.quantum_embeddings.iqp_embedding, num_qubits)

from scipy.optimize import minimize
from utils.error_functions import mean_square_error
from pennylane.optimize import (
    NesterovMomentumOptimizer,
    AdamOptimizer,
    GradientDescentOptimizer,
)

training_ratio = 0.8
"""

def minimisation(cost_function, params):
    result = minimize(
        cost_function,
        params,
        method="COBYLA",
        options={
            "rhobeg": 1000,
            "disp": True,  # To reduce the time of optimization
        },
    )
    return result.x


"""


def minimisation(cost_function, params):
    result = GradientDescentOptimizer(stepsize=100)
    new_params = params
    for _ in range(100):
        new_params, fct = result.step_and_cost(cost_function, new_params)
        print(new_params)
        print(fct)
    return new_params


def Bcel(predicted_labels, expirement_labels) -> float:
    av_pred = np.average(predicted_labels)
    av_test = np.average(expirement_labels)
    return -((av_test * np.log(av_pred)) + ((1 - av_test) * np.log(1 - av_pred)))


def cost(predicted_labels, expirement_labels) -> float:
    return -(utils.utils.get_score(predicted_labels, expirement_labels))


def new_cost(predicted_labels, expirement_labels):
    return -(utils.error_functions.accuracy(predicted_labels, expirement_labels))


score, predictions = qcnn.run(
    feature_vectors,
    labels,
    minimisation,
    error_function=new_cost,
    training_ratio=training_ratio,
)

training_period = int(len(labels) * training_ratio)

print("The score of the normal QCNN: ", score)
print("The predictions of the labels: ", predictions)
print("The true value of the labels: ", labels[training_period:])
