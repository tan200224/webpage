@echo off
echo Starting VAE Backend and React Frontend...

echo.
echo Starting Python Flask Backend...
start cmd /k python main.py

echo.
echo Starting React Frontend...
start cmd /k npm run dev

echo.
echo Services started! Access the app at http://localhost:5173
echo. 