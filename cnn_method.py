import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Plus juste, mais moins performant
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import numpy as np
from numpy.typing import NDArray


def build_model(input_shape):
    """
    Build a fully connected neural network model using Keras Sequential API.

    Parameters:
    input_shape (int): The number of input features for the first layer.

    Returns:
    Sequential: A Keras Sequential model.
    """
    model = models.Sequential()
    model.add(
        layers.Input(shape=(input_shape,))
    )  # Define input shape using Input layer
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dense(32, activation="relu"))
    model.add(layers.Dense(16, activation="relu"))
    model.add(layers.Dense(1, activation="sigmoid"))

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def train_and_evaluate(feature_vectors: NDArray[np.float_], labels: NDArray[np.int_]):
    """
    Train and evaluate the neural network model.

    Parameters:
    feature_vectors (NDArray[np.float_]): The matrix of the feature vectors
    labels          (NDArray[np.float_]): The vector of the set's labels

    Returns:
    float: The accuracy of the model on the test set.
    """

    # Load data
    X = feature_vectors
    y = labels

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Standardize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Build and train the model
    model = build_model(X_train.shape[1])
    model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)

    # Evaluate the model
    y_pred = model.predict(X_test) > 0.5
    accuracy = accuracy_score(y_test, y_pred)

    return accuracy
