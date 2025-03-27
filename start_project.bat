@echo off
echo Starting the ArchieProflio Synthetic CT Generator Application...

REM Check if we're in the right directory
IF NOT EXIST BackEnd (
    echo Error: BackEnd directory not found. Please run this script from the project root.
    goto :end
)

IF NOT EXIST FrontEnd (
    echo Error: FrontEnd directory not found. Please run this script from the project root.
    goto :end
)

echo.
echo Checking for required dependencies...

REM Verify Python is installed
python --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo Error: Python not found. Please install Python 3.8+ and try again.
    goto :end
)

REM Verify Node.js is installed
node --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo Error: Node.js not found. Please install Node.js and try again.
    goto :end
)

echo.
echo Starting Python Flask Backend...
cd BackEnd
start cmd /k python main.py

echo.
echo Starting React Frontend...
cd ../FrontEnd
start cmd /k npm run dev

echo.
echo Services started! 
echo Backend API is running at http://localhost:5000
echo Frontend application is running at http://localhost:5173
echo.
echo Use Ctrl+C in respective terminal windows to stop the servers.

:end
cd %~dp0 