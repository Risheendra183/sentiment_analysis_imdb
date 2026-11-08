import streamlit as st
import tensorflow as tf
import numpy as np
import json
import os

from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RNN Sentiment Analysis",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 Movie Review Sentiment Analysis")
st.write("Compare **SimpleRNN**, **LSTM**, and **GRU** on your review in real time.")

# ── Constants ─────────────────────────────────────────────────────────────────
VOCAB_SIZE  = 10000
MAX_LENGTH  = 200
BASE_DIR    = os.path.dirname(__file__)

MODEL_FILES = {
    "SimpleRNN": "rnn_sentiment.keras",
    "LSTM":      "lstm_sentiment.keras",
    "GRU":       "gru_sentiment.keras",
}

MODEL_INFO = {
    "SimpleRNN": "Lightweight & fast — can struggle with long-range dependencies.",
    "LSTM":      "Long Short-Term Memory — handles long sequences well via gating.",
    "GRU":       "Gated Recurrent Unit — faster than LSTM, usually similar accuracy.",
}


# ── Cached loaders ────────────────────────────────────────────────────────────
@st.cache_resource
def load_all_models():
    models = {}
    for name, filename in MODEL_FILES.items():
        path = os.path.join(BASE_DIR, filename)
        if os.path.exists(path):
            try:
                models[name] = tf.keras.models.load_model(path, compile=False)
            except Exception:
                # Fallback: try without compile flag for older keras files
                models[name] = tf.keras.models.load_model(path)
        else:
            models[name] = None
    return models


@st.cache_resource
def load_word_index():
    return imdb.get_word_index()


@st.cache_data
def load_results():
    path = os.path.join(BASE_DIR, "model_results.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


models     = load_all_models()
word_index = load_word_index()
results    = load_results()


# ── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess_text(text: str):
    words    = text.lower().split()
    sequence = []
    for word in words:
        index = word_index.get(word, 2) + 3
        if index >= VOCAB_SIZE:
            index = 2
        sequence.append(index)
    return pad_sequences(
        [sequence], maxlen=MAX_LENGTH, padding='post', truncating='post'
    )


# ── Sidebar — test-set performance ────────────────────────────────────────────
st.sidebar.header("📊 Test-Set Performance")

if results:
    best_model = max(results, key=lambda k: results[k]["test_accuracy"])
    for name in MODEL_FILES:
        if name in results:
            r     = results[name]
            crown = " 🏆" if name == best_model else ""
            st.sidebar.metric(
                label=f"{name}{crown}",
                value=f"{r['test_accuracy'] * 100:.2f}%",
                delta=f"Loss: {r['test_loss']:.4f}"
            )
        else:
            st.sidebar.metric(label=name, value="—", delta="Not evaluated yet")
else:
    st.sidebar.info(
        "No results yet.\n\n"
        "After training, run:\n```\npython evaluate_all.py\n```"
    )

st.sidebar.markdown("---")
st.sidebar.header("ℹ️ About the models")
for name, info in MODEL_INFO.items():
    st.sidebar.markdown(f"**{name}** — {info}")


# ── Comparison chart ──────────────────────────────────────────────────────────
plot_path = os.path.join(BASE_DIR, "comparison_plot.png")
if os.path.exists(plot_path):
    with st.expander("📈 Training comparison chart", expanded=False):
        st.image(plot_path, width='stretch')

st.markdown("---")


# ── Review input ──────────────────────────────────────────────────────────────
review = st.text_area(
    "✍️ Enter your movie review",
    height=120,
    placeholder="e.g. The acting was brilliant and the story kept me hooked throughout!"
)

col_btn, col_sel = st.columns([1, 3])
with col_btn:
    analyze = st.button("🔍 Analyze Sentiment", width='stretch')
with col_sel:
    selected_models = st.multiselect(
        "Models to run",
        options=list(MODEL_FILES.keys()),
        default=list(MODEL_FILES.keys())
    )


# ── Prediction & results ──────────────────────────────────────────────────────
if analyze:
    if not review.strip():
        st.warning("Please enter a review first.")
    elif not selected_models:
        st.warning("Select at least one model.")
    else:
        processed = preprocess_text(review)

        st.markdown("---")
        st.subheader("Results")

        cols = st.columns(len(selected_models))

        for col, name in zip(cols, selected_models):
            with col:
                st.markdown(f"### {name}")

                if models[name] is None:
                    st.error(
                        f"`{MODEL_FILES[name]}` not found.\n\n"
                        f"Run `python train{'_' + name.lower() if name != 'SimpleRNN' else ''}.py` first."
                    )
                else:
                    prob       = float(models[name].predict(processed, verbose=0)[0][0])
                    positive   = prob >= 0.5
                    sentiment  = "Positive 😊" if positive else "Negative 😞"
                    confidence = prob * 100 if positive else (1 - prob) * 100
                    color      = "green" if positive else "red"

                    st.markdown(
                        f"<h4 style='color:{color}'>{sentiment}</h4>",
                        unsafe_allow_html=True
                    )
                    st.progress(float(prob if positive else 1 - prob))
                    st.metric("Confidence", f"{confidence:.1f}%")

                    if name in results:
                        st.caption(
                            f"Test accuracy: {results[name]['test_accuracy']*100:.2f}%"
                        )
                    st.caption(MODEL_INFO[name])
