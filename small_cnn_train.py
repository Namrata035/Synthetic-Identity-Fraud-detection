# small_cnn_train.py
"""
Train a small CNN on MNIST and save the model.
Usage:
    python small_cnn_train.py
Outputs:
    - model/mnist_cnn.h5
    - model/train_history.npy  (loss/acc history for quick inspection)
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

# Hyperparams
BATCH_SIZE = 128
EPOCHS = 5
LEARNING_RATE = 1e-3
VALIDATION_SPLIT = 0.1
INPUT_SHAPE = (28, 28, 1)

def load_preprocess():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # normalize to [0,1]
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    # add channel dim
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)
    return (x_train, y_train), (x_test, y_test)

def build_model(input_shape):
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2,2)),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D((2,2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(10, activation='softmax')
    ])
    return model

def main():
    (x_train, y_train), (x_test, y_test) = load_preprocess()
    model = build_model(INPUT_SHAPE)
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()

    history = model.fit(
        x_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=VALIDATION_SPLIT,
        verbose=2
    )

    model_path = os.path.join(MODEL_DIR, "mnist_cnn.h5")
    model.save(model_path)
    print(f"[✓] Saved model to {model_path}")

    # save small history (numpy) for quick plotting/inspection if desired
    hist_path = os.path.join(MODEL_DIR, "train_history.npy")
    np.save(hist_path, history.history)
    print(f"[✓] Saved training history to {hist_path}")

if __name__ == "__main__":
    main()
