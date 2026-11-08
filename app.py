import streamlit as st
import tensorflow as tf
import numpy as np

from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

st.set_page_config(
    page_title="RNN Sentiment Analysis",
    page_icon="🎬"
)

st.title("Movie Review Sentiment Analysis")

st.write(
    "Enter a movie review to predict its sentiment."
)

VOCAB_SIZE = 10000
MAX_LENGTH = 200

@st.cache_resource
def load_model():

    return tf.keras.models.load_model(
        "rnn_sentiment.keras"
    )

model = load_model()

word_index = imdb.get_word_index()

review = st.text_area(
    "Enter your movie review"
)

def preprocess_text(text):

    words = text.lower().split()

    sequence = []

    for word in words:

        index = word_index.get(word, 2) + 3

        if index >= VOCAB_SIZE:
            index = 2

        sequence.append(index)

    return pad_sequences(
        [sequence],
        maxlen=MAX_LENGTH,
        padding='post',
        truncating='post'
    )

if st.button("Analyze Sentiment"):

    if review.strip():

        processed = preprocess_text(review)

        prediction = model.predict(
            processed,
            verbose=0
        )[0][0]

        if prediction >= 0.5:

            st.success("Positive Sentiment")

            st.write(
                f"Model score: {prediction * 100:.2f}%"
            )

        else:

            st.error("Negative Sentiment")

            st.write(
                f"Model score: {(1-prediction) * 100:.2f}%"
            )

    else:

        st.warning("Please enter a review.")