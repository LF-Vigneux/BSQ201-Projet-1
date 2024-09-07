import numpy as np
from sklearn.svm import SVC
from numpy.typing import NDArray


def get_kernel_prediction(
    qkernel: callable,
    feature_vectors: NDArray[np.float_],
    labels: NDArray[np.int_],
    training_period: int,
):
    training_vectors = feature_vectors[:training_period, :]
    testing_vecors = feature_vectors[training_period:, :]
    training_labels = feature_vectors[:training_period]
    testing_labels = feature_vectors[training_period:]

    model = SVC(kernel=qkernel)
    model.fit(training_vectors, training_labels)

    score = model.score(testing_vecors, testing_labels)
    predictions = model.predict(testing_vecors)

    return score, predictions
