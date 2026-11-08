@echo off
cd /d "%~dp0"
echo Starting Streamlit App...
python -m streamlit run app.py
pause

