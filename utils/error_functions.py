import numpy as np
from numpy.typing import NDArray
from utils.utils import get_accuracies
from statistics import stdev


def mean_square_error(
    predicted_labels: NDArray[np.float_], expirement_labels: NDArray[np.float_]
):
    total = 0
    for true_value, predicted in zip(predicted_labels, expirement_labels):
        total += (predicted - true_value) ** 2
    return total / (2 * len(predicted_labels))


def normalized_mean_square_error(
    predicted_labels: NDArray[np.float_], expirement_labels: NDArray[np.float_]
):
    total_num = 0
    total_denom = 0
    for true_value, predicted in zip(predicted_labels, expirement_labels):
        total_num += (predicted - true_value) ** 2
        total_denom += true_value**2
    return total_num / total_denom


def normalized_root_mean_square_error(
    predicted_labels: NDArray[np.float_], expirement_labels: NDArray[np.float_]
):
    total = 0
    for true_value, predicted in zip(predicted_labels, expirement_labels):
        total += (predicted - true_value) ** 2
    return np.sqrt((1 / len(predicted_labels)) * total) / stdev(predicted_labels)


# These functions are inspired by the pulsar paper
def accuracy(
    predicted_labels: NDArray[np.float_], expirement_labels: NDArray[np.float_]
) -> float:
    true_positive, false_positive, true_negative, false_negative = get_accuracies(
        predicted_labels, expirement_labels
    )

    print(
        (true_positive + true_negative)
        / (true_positive + false_positive + true_negative + false_negative)
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
    # If there is about to be a division by 0, return -5 as a error message.
    if true_positive + false_negative == 0:
        return -5
    return true_positive / (true_positive + false_negative)


def specifity(
    predicted_labels: NDArray[np.float_], expirement_labels: NDArray[np.float_]
) -> float:
    true_positive, false_positive, true_negative, false_negative = get_accuracies(
        predicted_labels, expirement_labels
    )
    # If there is about to be a division by 0, return -5 as a error message.
    if true_negative + false_positive == 0:
        return -5
    return true_negative / (true_negative + false_positive)


def precision(
    predicted_labels: NDArray[np.float_], expirement_labels: NDArray[np.float_]
) -> float:
    true_positive, false_positive, true_negative, false_negative = get_accuracies(
        predicted_labels, expirement_labels
    )
    # If there is no positives predicted, return -5 as an error message.
    if true_positive + false_positive == 0:
        return -5
    return true_positive / (true_positive + false_positive)


def negative_prediction_value(
    predicted_labels: NDArray[np.float_], expirement_labels: NDArray[np.float_]
) -> float:
    true_positive, false_positive, true_negative, false_negative = get_accuracies(
        predicted_labels, expirement_labels
    )
    # If there is no negatives predicted, return -5 as an error message.
    if true_negative + false_negative == 0:
        return -5
    return true_negative / (true_negative + false_negative)


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
    recall_result = recall(predicted_labels, expirement_labels)
    specifity_result = specifity(predicted_labels, expirement_labels)
    # If there was a division by 0 in the recall or specifity, return -5 as an error message.
    if recall_result == -5 or specifity_result == -5:
        return -5

    return recall_result + specifity_result - 1
