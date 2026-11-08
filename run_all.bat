@echo off
echo =============================================
echo   RNN Sentiment Analysis - Full Run Script
echo =============================================
echo.

cd /d "%~dp0"

echo Removing old model files to avoid version mismatch...
if exist models\rnn_sentiment.keras  del /f models\rnn_sentiment.keras
if exist models\lstm_sentiment.keras del /f models\lstm_sentiment.keras
if exist models\gru_sentiment.keras  del /f models\gru_sentiment.keras
if exist models\model_results.json   del /f models\model_results.json
echo Done.
echo.

echo [1/4] Training SimpleRNN...
python scripts\train.py
echo.

echo [2/4] Training LSTM...
python scripts\train_lstm.py
echo.

echo [3/4] Training GRU...
python scripts\train_gru.py
echo.

echo [4/4] Evaluating all models and generating comparison metrics...
python scripts\evaluate_all.py
echo.

echo =============================================
echo   All done! Launching Streamlit app...
echo =============================================
python -m streamlit run app.py

pause
