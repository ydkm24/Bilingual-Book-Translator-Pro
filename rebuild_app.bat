@echo off
setlocal
echo ==================================================
echo   Bilingual Book Translator Pro: Smart Rebuilder
echo ==================================================
echo.

:: 1. Detect Python
echo [1/3] Checking Python environment...
py --version >nul 2>&1
if %errorlevel% neq 0 (
    python --version >nul 2>&1
    if %errorlevel% neq 0 (
        echo [ERROR] Python not found! Please install Python 3.11+.
        pause
        exit /b 1
    )
    set PY_CMD=python
) else (
    set PY_CMD=py
)
echo Found: %PY_CMD%

:: 2. Update Environment
echo [2/3] Updating libraries and build tools...
%PY_CMD% -m pip install --upgrade pip
%PY_CMD% -m pip install pyinstaller
if exist requirements.txt (
    %PY_CMD% -m pip install -r requirements.txt
)

:: 3. Build EXE
echo.
echo [3/3] Building Windows EXE (Grab a coffee, this takes ~2-4 minutes)...
echo Target Spec: pdf_translator.spec
%PY_CMD% -m PyInstaller --noconfirm pdf_translator.spec

if %errorlevel% neq 0 (
    echo.
    echo [CRITICAL ERROR] Build failed! Check the messages above.
) else (
    echo.
    echo ==================================================
    echo   BUILD SUCCESSFUL!
    echo   New EXE is at: dist\BilingualBookTranslatorPro.exe
    echo ==================================================
)

pause
