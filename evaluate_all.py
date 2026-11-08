"""
evaluate_all.py
Run this ONCE after training all 3 models to generate model_results.json
which the Streamlit app uses for the sidebar performance metrics.

Usage:
    python evaluate_all.py
"""

import tensorflow as tf
import json
import os

from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

VOCAB_SIZE = 10000
MAX_LENGTH = 200
BASE_DIR   = os.path.dirname(__file__)

MODEL_FILES = {
    "SimpleRNN": "rnn_sentiment.keras",
    "LSTM":      "lstm_sentiment.keras",
    "GRU":       "gru_sentiment.keras",
}

# Load dataset
print("Loading IMDB test set...")
(_, _), (X_test, y_test) = imdb.load_data(num_words=VOCAB_SIZE)
X_test = pad_sequences(
    X_test,
    maxlen=MAX_LENGTH,
    padding='post',
    truncating='post'
)

results = {}

for name, filename in MODEL_FILES.items():
    path = os.path.join(BASE_DIR, filename)
    if not os.path.exists(path):
        print(f"[{name}] Model not found — skipping. Run the training script first.")
        continue

    print(f"\nEvaluating {name}...")
    model = tf.keras.models.load_model(path, compile=False)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

    results[name] = {
        "test_loss":     float(test_loss),
        "test_accuracy": float(test_accuracy)
    }

    print(f"  Test Accuracy : {test_accuracy:.4f}")
    print(f"  Test Loss     : {test_loss:.4f}")

# Save
results_path = os.path.join(BASE_DIR, "model_results.json")
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved → {results_path}")

# Summary table
print("\n" + "="*45)
print(f"{'Model':<12} {'Test Accuracy':>15} {'Test Loss':>12}")
print("="*45)
for name, r in results.items():
    print(f"{name:<12} {r['test_accuracy']:>14.4f}  {r['test_loss']:>12.4f}")
print("="*45)

if results:
    best = max(results, key=lambda k: results[k]['test_accuracy'])
    print(f"\n🏆  Best model: {best}  ({results[best]['test_accuracy']*100:.2f}%)")

    # Generate comparison plot for the Streamlit app
    try:
        import matplotlib.pyplot as plt

        models = list(results.keys())
        accs = [results[m]["test_accuracy"] * 100 for m in models]
        losses = [results[m]["test_loss"] for m in models]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        colors = ['#4A90E2', '#50E3C2', '#F5A623'][:len(models)]

        bars1 = ax1.bar(models, accs, color=colors, width=0.5)
        ax1.set_title("Test Accuracy (%)", fontweight='bold')
        ax1.set_ylim(0, 100)
        ax1.grid(axis='y', linestyle='--', alpha=0.5)
        for bar, acc in zip(bars1, accs):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5, f"{acc:.2f}%", ha='center', fontweight='bold')

        bars2 = ax2.bar(models, losses, color=colors, width=0.5)
        ax2.set_title("Test Loss", fontweight='bold')
        ax2.grid(axis='y', linestyle='--', alpha=0.5)
        for bar, loss in zip(bars2, losses):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f"{loss:.4f}", ha='center', fontweight='bold')

        plt.tight_layout()
        plot_path = os.path.join(BASE_DIR, "comparison_plot.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Generated comparison plot → {plot_path}")
    except Exception as e:
        print(f"Could not generate plot: {e}")

