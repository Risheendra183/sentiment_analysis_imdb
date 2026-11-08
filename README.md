# 🎬 IMDB Movie Review Sentiment Analysis: SimpleRNN vs. LSTM vs. GRU

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://movie-sentimentanalysis-imdb.streamlit.app/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end Natural Language Processing (NLP) deep learning project that classifies movie review sentiments as **Positive 😊** or **Negative 😞**. 

This repository implements, benchmarks, and provides interactive side-by-side comparisons of three recurrent neural network architectures: **SimpleRNN**, **LSTM (Long Short-Term Memory)**, and **GRU (Gated Recurrent Unit)** trained on the **IMDB Reviews dataset**.

---

## 🌐 Live Web Application

Try out the live interactive app directly in your browser:
👉 **[movie-sentimentanalysis-imdb.streamlit.app](https://movie-sentimentanalysis-imdb.streamlit.app/)**

### 📸 Application Highlights
- **Interactive Sentiment Prediction**: Input any custom movie review and get real-time sentiment predictions.
- **Multi-Model Comparison**: Run your review across SimpleRNN, LSTM, and GRU simultaneously to inspect prediction differences and confidence percentages.
- **Visual Confidence Gauges**: Dynamic progress indicators showcasing model certainty.
- **Sidebar Benchmark Dashboard**: Displays test accuracy, loss metrics, and architecture summaries.
- **Training Comparison Analytics**: Embedded visual comparison charts for test loss and accuracy.

---

## 📊 Model Performance Benchmark

All models were evaluated on the 25,000-sample IMDB test dataset using identical vocabulary limits (`10,000` words) and sequence lengths (`200` tokens):

| Model | Architecture | Test Accuracy | Test Loss | Status |
| :--- | :--- | :---: | :---: | :---: |
| **SimpleRNN** | Embedding (64) + SimpleRNN (64) + Dense (32) + Dropout (0.5) + Sigmoid | **77.86%** | `0.6193` | Baseline |
| **LSTM** | Embedding (64) + LSTM (64) + Dense (32) + Dropout (0.5) + Sigmoid | **81.22%** | `0.5618` | Strong Long-Range Handling |
| **GRU** | Embedding (64) + GRU (64) + Dense (32) + Dropout (0.5) + Sigmoid | **85.05%** | `0.4480` | 🏆 **Best Performer** |

### Key Takeaways
1. **SimpleRNN** suffers from the vanishing gradient problem when handling sequence lengths of 200 tokens, achieving an accuracy of ~77.86%.
2. **LSTM** resolves the vanishing gradient issue through input, forget, and output gating mechanisms, jumping to 81.22% accuracy.
3. **GRU** provides the best generalization on this dataset with 85.05% accuracy and the lowest test loss (0.4480), converging faster thanks to its streamlined reset and update gate design.

---

## 🧠 Model Architectures

Each model shares a standardized architecture to ensure fair comparison:

```text
Input Sequence (200 tokens)
       │
       ▼
Embedding Layer (Vocab: 10,000, Output Dim: 64, mask_zero=True)
       │
       ▼
Recurrent Layer (SimpleRNN / LSTM / GRU with 64 units)
       │
       ▼
Dense Hidden Layer (32 units, ReLU activation)
       │
       ▼
Dropout Layer (Rate = 0.5)
       │
       ▼
Dense Output Layer (1 unit, Sigmoid activation)
```

- **Optimizer**: Adam
- **Loss Function**: Binary Crossentropy
- **Batch Size**: 64
- **Epochs**: 5 (with 20% validation split)

---

## 📁 Repository Structure

```text
sentiment_analysis_imdb/
├── app.py                   # Streamlit web application
├── train.py                 # SimpleRNN training script
├── train_lstm.py            # LSTM training script
├── train_gru.py             # GRU training script
├── evaluate_all.py          # Unified evaluation & comparison plot generator
├── run_all.bat              # One-click Windows batch runner to train & launch
├── requirements.txt         # Python dependencies
├── model_results.json       # Benchmark metrics for app sidebar
├── rnn_sentiment.keras      # Serialized SimpleRNN model
├── lstm_sentiment.keras     # Serialized LSTM model
├── gru_sentiment.keras      # Serialized GRU model
├── comparison_plot.png      # Side-by-side metric visualizations
└── README.md                # Project documentation
```

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/Risheendra183/sentiment_analysis_imdb.git
cd sentiment_analysis_imdb
```

### 2. Create and Activate a Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch the Streamlit App
```bash
streamlit run app.py
```
The application will open automatically at `http://localhost:8501`.

---

## 🏋️ Training & Evaluation

If you wish to retrain the models from scratch:

### Retrain Models
```bash
# 1. Train SimpleRNN
python train.py

# 2. Train LSTM
python train_lstm.py

# 3. Train GRU
python train_gru.py
```

### Evaluate All Models & Update Benchmark Results
```bash
python evaluate_all.py
```
This script evaluates each `.keras` model on the IMDB test set, updates `model_results.json`, and generates `comparison_plot.png`.

### Automated Pipeline (Windows)
Run `run_all.bat` to clean old checkpoints, train all 3 networks sequentially, evaluate results, and start the Streamlit server:
```cmd
run_all.bat
```

---

## 🛠️ Built With

- **[Streamlit](https://streamlit.io/)** — Web application and user interface
- **[TensorFlow / Keras](https://www.tensorflow.org/)** — Model design, training, and recurrent neural network layers
- **[NumPy](https://numpy.org/)** — Numerical operations and sequence transformations
- **[Matplotlib](https://matplotlib.org/)** — Training history and benchmark visualization plots
- **[IMDB Movie Reviews Dataset](https://ai.stanford.edu/~amaas/data/sentiment/)** — 50,000 labeled movie reviews

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

