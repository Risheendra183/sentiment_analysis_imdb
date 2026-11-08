import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import json

from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Configuration
VOCAB_SIZE = 10000
MAX_LENGTH = 200

# Load dataset
(X_train, y_train), (X_test, y_test) = imdb.load_data(
    num_words=VOCAB_SIZE
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

print("First review:", X_train[0])
print("First label:", y_train[0])

X_train = pad_sequences(
    X_train,
    maxlen=MAX_LENGTH,
    padding='post',
    truncating='post'
)

X_test = pad_sequences(
    X_test,
    maxlen=MAX_LENGTH,
    padding='post',
    truncating='post'
)

print(X_train.shape)
print(X_test.shape)



model = tf.keras.Sequential([

    tf.keras.layers.Input(
        shape=(MAX_LENGTH,)
    ),

    tf.keras.layers.Embedding(
        input_dim=VOCAB_SIZE,
        output_dim=64,
        mask_zero=True
    ),

    tf.keras.layers.GRU(
        64
    ),

    tf.keras.layers.Dense(
        32,
        activation='relu'
    ),

    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(
        1,
        activation='sigmoid'
    )
])

model.summary()

model.compile(

    optimizer='adam',

    loss='binary_crossentropy',

    metrics=['accuracy']
)


history = model.fit(

    X_train,
    y_train,

    epochs=5,

    batch_size=64,

    validation_split=0.2
)



plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)

plt.plot(
    history.history['accuracy'],
    label='Training Accuracy'
)

plt.plot(
    history.history['val_accuracy'],
    label='Validation Accuracy'
)

plt.title('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)

plt.plot(
    history.history['loss'],
    label='Training Loss'
)

plt.plot(
    history.history['val_loss'],
    label='Validation Loss'
)

plt.title('Loss')
plt.legend()

save_path = os.path.join(os.path.dirname(__file__), "gru_sentiment.keras")
model.save(save_path)
print(f"Model saved successfully to {save_path}")

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test
)

print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

results_path = os.path.join(os.path.dirname(__file__), "model_results.json")
results = {}
if os.path.exists(results_path):
    with open(results_path) as f:
        results = json.load(f)

results["GRU"] = {
    "test_loss": float(test_loss),
    "test_accuracy": float(test_accuracy)
}

with open(results_path, "w") as f:
    json.dump(results, f, indent=2)

print("Saved results to model_results.json")

plt.savefig(os.path.join(os.path.dirname(__file__), "gru_history.png"))
plt.close()

