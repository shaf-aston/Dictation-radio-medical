@echo off
call .venv\Scripts\activate 2>nul || echo .venv not found, using global python
python -m src.ui.web_app
pause
