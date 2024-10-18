"""
File containing the fonction to run the svm algorithm.
This code is from:
    NAVLANI Avinash (December 27th, 2019). "Support Vector Machines with Scikit-learn Tutorial", 
datacamp, [Online], https://www.datacamp.com/tutorial/svm-classification-scikit-learn-python, 
(Consulted on October 17th 2024).
"""

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from typing import Tuple
from numpy.typing import NDArray


def svm_run(
    feature_vectors, labels, training_ratio=0.8, kernel="rbf"
) -> Tuple[float, NDArray]:
    """
    Train and evaluate the neural network model.

    Parameters:
    - feature_vectors (NDArray[np.float_]): The matrix of the feature vectors
    - labels (NDArray[np.float_]): The vector of the set's labels
    - training_ratio (float=0.8): The ratio of the number of feature vectors used for training over the total number of feature vectors.

    Returns:
    tuple[float, NDArray]:     - The number of correctly predicted labels.
                               - The prediction labels of the testing feature vectors.
    """

    # Split the data into training and test sets
    feature_vectors_train, feature_vectors_test, labels_train, labels_test = (
        train_test_split(
            feature_vectors, labels, train_size=training_ratio, random_state=42
        )
    )

    # Standardize the data
    scaler = StandardScaler()
    feature_vectors_train_scaled = scaler.fit_transform(feature_vectors_train)
    feature_vectors_test_scaled = scaler.transform(feature_vectors_test)

    # Train a Support Vector Classifier
    svm = SVC(
        kernel=kernel
    )  # You can also experiment with other kernels like 'linear' or 'poly'
    svm.fit(feature_vectors_train_scaled, labels_train)

    # Make predictions
    labels_pred = svm.predict(feature_vectors_test_scaled)

    # Evaluate the model
    accuracy = accuracy_score(labels_test, labels_pred)

    return accuracy, labels_pred
