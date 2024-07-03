import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist

# Function to train the model
def train_model(model_file):
    # Loading the MNIST dataset
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Preprocessing the data
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    # Creating the model
    model = Sequential()
    model.add(Flatten(input_shape=(28, 28)))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    # Compiling the model
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Training the model
    model.fit(X_train, y_train, epochs=10, validation_split=0.2)

    # Evaluating the model on test data
    _, accuracy = model.evaluate(X_test, y_test)
    print(f'Model accuracy: {accuracy}')

    # Saving the entire model in keras format
    model.save(model_file)

# Entry point to train the model if executed directly
if __name__ == "__main__":
    model_file = 'model.keras'
    train_model(model_file)