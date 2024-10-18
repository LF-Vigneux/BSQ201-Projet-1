"""
This file contains the functions of the CNN method. The cnn_run is the main function to run this algorithm.
This was inspired by:
    KERAS GOOGLE GROUP [n.d.]. "Getting started with the Keras Sequential model", 
Keras Documentation, [Online], https://faroit.com/keras-docs/1.0.1/getting-started/sequential-model-guide/, 
(Consulted on October 17th 2024).
"""

import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import numpy as np
from numpy.typing import NDArray


def build_model(input_shape: int) -> models.Sequential:
    """
    Build a fully connected neural network model using Keras Sequential API.

    Parameters:
    - input_shape (int): The number of input features for the first layer.

    Returns:
    models.Sequential : A compiled Keras Sequential model ready for training.
    """
    model = models.Sequential()
    model.add(layers.Input(shape=(input_shape,)))

    # Define input shape using Input layer
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dense(32, activation="relu"))
    model.add(layers.Dense(16, activation="relu"))
    model.add(layers.Dense(1, activation="sigmoid"))

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def cnn_run(
    feature_vectors: NDArray[np.float_],
    labels: NDArray[np.int_],
    training_ratio: float,
    batch_size: int,
) -> tuple[float, NDArray]:
    """
    Train and evaluate the neural network model.

    Parameters:
    - feature_vectors (NDArray[np.float_]): The matrix of the feature vectors
    - labels          (NDArray[np.float_]): The vector of the set's labels
    - training_ratio  (float)             : The ratio of the number of feature vectors used for training over the total number of feature vectors.
    - batch_size      (int)               : The batch size used during model training.

    Returns:
    tuple[float, NDArray]:     - The number of correctly predicted labels.
                               - The prediction labels of the testing feature vectors.
                               - The labels used during the testing of the feature vectors.
    """

    labels = (labels + 1) / 2

    # Split the dataset into training and testing sets
    feature_vectors_train, feature_vectors_test, labels_train, labels_test = (
        train_test_split(
            feature_vectors, labels, train_size=training_ratio, random_state=42
        )
    )

    # Standardize the data
    scaler = StandardScaler()
    feature_vectors_train = scaler.fit_transform(feature_vectors_train)
    feature_vectors_test = scaler.transform(feature_vectors_test)

    # Build and train the model
    model = build_model(feature_vectors_train.shape[1])
    model.fit(
        feature_vectors_train, labels_train, epochs=20, batch_size=batch_size, verbose=0
    )

    # Evaluate the model
    labels_pred = model.predict(feature_vectors_test, verbose=0) > 0.5

    accuracy = accuracy_score(labels_test, labels_pred)

    # Transform the labels
    labels_pred=np.reshape(labels_pred,np.shape(labels_pred)[0])
    labels_test=np.reshape(labels_test,np.shape(labels_pred)[0])
    
    labels_pred = np.where(labels_pred, 1, -1)
    labels_test = np.where(labels_test, 1, -1)

    

    return accuracy, labels_pred, labels_test
