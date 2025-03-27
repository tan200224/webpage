@echo off
echo Installing requirements for ArchieProflio Synthetic CT Generator...

REM Check for Python
python --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo Error: Python not found. Please install Python 3.8+ and try again.
    goto :end
)

REM Check for Node.js
node --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo Error: Node.js not found. Please install Node.js and try again.
    goto :end
)

echo.
echo Installing Python dependencies...
cd BackEnd
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo.
echo Installing Node.js dependencies...
cd ../FrontEnd
npm install

echo.
echo Installation complete!
echo You can now run the application using start_project.bat

:end
cd %~dp0 