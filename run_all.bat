@echo off
echo =============================================
echo   RNN Sentiment Analysis - Full Run Script
echo =============================================
echo.

cd /d "C:\Users\user\Downloads\RNN project\movie_sentiment_analysis"

echo Removing old model files to avoid version mismatch...
if exist rnn_sentiment.keras  del /f rnn_sentiment.keras
if exist lstm_sentiment.keras del /f lstm_sentiment.keras
if exist gru_sentiment.keras  del /f gru_sentiment.keras
if exist model_results.json   del /f model_results.json
echo Done.
echo.

echo [1/4] Training SimpleRNN...
C:\Python312\python.exe train.py
echo.

echo [2/4] Training LSTM...
C:\Python312\python.exe train_lstm.py
echo.

echo [3/4] Training GRU...
C:\Python312\python.exe train_gru.py
echo.

echo [4/4] Evaluating all models and generating comparison metrics...
C:\Python312\python.exe evaluate_all.py
echo.

echo =============================================
echo   All done! Launching Streamlit app...
echo =============================================
C:\Python312\python.exe -m streamlit run app.py

pause
