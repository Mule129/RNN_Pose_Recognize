import tensorflow as tf
from keras import (
    models,
    layers
)
from keras.models import (
    load_model,
    save_model
)
from sklearn.model_selection import train_test_split

from typing import Any
from numpy import ndarray

from models.handModel import HandModel


class HandFit:
    def __init__(self, x_data, y_data):
        self.model = models.Sequential(
            [
                layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 3), activation='relu'),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 3), activation='relu'),

                layers.Flatten(),
                layers.Dense(64, activation='relu'),
                layers.Dense(10)
            ]
        )
        self.train_labels = ["Rock", "Paper", "Scissors"]

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x_data, y_data)

    def fit_hand_model(self):
        self.model.compile(
            optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )

        history = self.model.fit(
            self.x_train, self.train_labels, epochs=10,
            validation_data=(self.x_test, self.train_labels)
        )

    def get_model(self):
        return self.model


if __name__ == "__main__":
    fit_model = HandFit()

    print(fit_model.get_model().summary())
