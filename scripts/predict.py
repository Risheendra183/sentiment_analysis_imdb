import os
import tensorflow as tf
import numpy as np

from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

VOCAB_SIZE   = 10000
MAX_LENGTH   = 200
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
MODELS_DIR   = os.path.join(PROJECT_ROOT, "models")

# Check models/ folder first, fallback to root or current dir
model_path = os.path.join(MODELS_DIR, "rnn_sentiment.keras")
if not os.path.exists(model_path):
    model_path = os.path.join(PROJECT_ROOT, "rnn_sentiment.keras")
if not os.path.exists(model_path):
    model_path = "rnn_sentiment.keras"

model = tf.keras.models.load_model(model_path)

word_index = imdb.get_word_index()

def preprocess_text(text):

    words = text.lower().split()

    sequence = []

    for word in words:

        index = word_index.get(word, 2) + 3

        if index >= VOCAB_SIZE:
            index = 2

        sequence.append(index)

    padded = pad_sequences(
        [sequence],
        maxlen=MAX_LENGTH,
        padding='post',
        truncating='post'
    )

    return padded


def predict_sentiment(text):

    processed = preprocess_text(text)

    probability = model.predict(
        processed,
        verbose=0
    )[0][0]

    if probability >= 0.5:

        print("Sentiment: Positive")

        print(
            f"Confidence: {probability * 100:.2f}%"
        )

    else:

        print("Sentiment: Negative")

        print(
            f"Confidence: {(1-probability) * 100:.2f}%"
        )


predict_sentiment(
    "The movie was amazing and wonderful"
)
