import numpy as np
from numpy.typing import NDArray
from utils import get_accuracies
from statistics import stdev


def mean_square_error(
    predicted_labels: NDArray[np.float_], expirement_labels: NDArray[np.float_]
) -> float:
    total = 0
    for true_value, predicted in zip(predicted_labels, expirement_labels):
        total += (predicted - true_value) ** 2
    return total / (2 * len(predicted_labels))


# Les deux prochaines viennent de mon stage mais pourraient être interessantes à tester
def normalized_mean_square_error(
    predicted_labels: NDArray[np.float_], expirement_labels: NDArray[np.float_]
) -> float:
    total_num = 0
    total_denom = 0
    for true_value, predicted in zip(predicted_labels, expirement_labels):
        total_num += (predicted - true_value) ** 2
        total_denom += true_value**2
    return total_num / total_denom


def normalized_root_mean_square_error(
    predicted_labels: NDArray[np.float_], expirement_labels: NDArray[np.float_]
) -> float:
    total = 0
    for true_value, predicted in zip(predicted_labels, expirement_labels):
        total += (predicted - true_value) ** 2
    return np.sqrt((1 / len(predicted_labels)) * total) / stdev(predicted_labels)


# Ces errors functions sont inspirées du papier de pulsar
def accuracy(
    predicted_labels: NDArray[np.float_], expirement_labels: NDArray[np.float_]
) -> float:
    true_positive, false_positive, true_negative, false_negative = get_accuracies(
        predicted_labels, expirement_labels
    )
    return (true_positive + true_negative) / (
        true_positive + false_positive + true_negative + false_negative
    )


def recall(
    predicted_labels: NDArray[np.float_], expirement_labels: NDArray[np.float_]
) -> float:
    true_positive, false_positive, true_negative, false_negative = get_accuracies(
        predicted_labels, expirement_labels
    )
    return true_positive / (true_positive + false_negative)


def specifity(
    predicted_labels: NDArray[np.float_], expirement_labels: NDArray[np.float_]
) -> float:
    true_positive, false_positive, true_negative, false_negative = get_accuracies(
        predicted_labels, expirement_labels
    )
    return true_negative / (true_negative + false_positive)


def precision(
    predicted_labels: NDArray[np.float_], expirement_labels: NDArray[np.float_]
) -> float:
    true_positive, false_positive, true_negative, false_negative = get_accuracies(
        predicted_labels, expirement_labels
    )
    return true_positive / (true_positive + false_positive)


def negative_prediction_value(
    predicted_labels: NDArray[np.float_], expirement_labels: NDArray[np.float_]
) -> float:
    true_positive, false_positive, true_negative, false_negative = get_accuracies(
        predicted_labels, expirement_labels
    )
    return true_negative / (true_negative + false_negative)


# Peut-être ceux-là optimiser les autres fonctions pour appeler juste une fois la boucle des true positive/negative


def balanced_accuracy(
    predicted_labels: NDArray[np.float_], expirement_labels: NDArray[np.float_]
) -> float:

    return 0.5 * (
        recall(predicted_labels, expirement_labels)
        + specifity(predicted_labels, expirement_labels)
    )


def geometric_mean(
    predicted_labels: NDArray[np.float_], expirement_labels: NDArray[np.float_]
) -> float:

    return np.sqrt(
        recall(predicted_labels, expirement_labels)
        * specifity(predicted_labels, expirement_labels)
    )


def informedness(
    predicted_labels: NDArray[np.float_], expirement_labels: NDArray[np.float_]
) -> float:

    return (
        recall(predicted_labels, expirement_labels)
        + specifity(predicted_labels, expirement_labels)
        - 1
    )
