@echo off
setlocal
cd /d "%~dp0"
title CleanCrop - YOLO label review
set PYTHONIOENCODING=utf-8
set HF_HUB_DISABLE_SYMLINKS_WARNING=1

rem ---- First run: create the Python environment -------------------------
if not exist ".venv\Scripts\python.exe" (
    echo First-time setup: installing CleanCrop ^(this takes a few minutes^)...
    python -m venv .venv
    if errorlevel 1 (
        echo.
        echo Python 3.9+ is required. Install it from https://www.python.org/downloads/ and try again.
        pause
        exit /b 1
    )
    .venv\Scripts\python.exe -m pip install --upgrade pip
    where nvidia-smi >nul 2>nul
    if errorlevel 1 (
        .venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
    ) else (
        .venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu124
    )
    .venv\Scripts\python.exe -m pip install -r requirements.txt
)

rem ---- Which dataset? (drag a folder onto this file, or type the path) ---
set "DATASET=%~1"
if "%DATASET%"=="" (
    echo.
    echo  Drag your YOLO dataset folder onto CleanCrop.bat, or paste its path below.
    echo  ^(The folder that contains data.yaml and the images / labels folders^)
    echo.
    set /p "DATASET=Dataset folder: "
)
set DATASET=%DATASET:"=%
if "%DATASET:~-1%"=="\" set "DATASET=%DATASET:~0,-1%"
if not exist "%DATASET%\" (
    echo Folder not found: %DATASET%
    pause
    exit /b 1
)
for %%I in ("%DATASET%") do set "NAME=%%~nxI"

rem ---- Continue an earlier review, or analyse (again) --------------------
if exist "curation_output\%NAME%\curation_report.json" (
    echo.
    echo  An earlier analysis of "%NAME%" exists.
    echo    [1] Continue reviewing it
    echo    [2] Analyse again ^(after you changed or retrained the dataset^)
    echo    [3] Export flagged boxes as images into a folder to verify
    echo    [4] Copy flagged images + labels into a folder for X-AnyLabeling
    echo    [5] Copy fixed labels back into the dataset ^(after X-AnyLabeling^)
    choice /c 12345 /n /m " Choose 1-5: "
    if errorlevel 5 goto syncback
    if errorlevel 4 goto labeling
    if errorlevel 3 goto export
    if errorlevel 2 goto analyse
    .venv\Scripts\python.exe dataset_curator_yolo.py "%DATASET%" --serve-only
    goto end
)
goto analyse

:labeling
.venv\Scripts\python.exe dataset_curator_yolo.py "%DATASET%" --export-labeling
goto end

:syncback
.venv\Scripts\python.exe dataset_curator_yolo.py "%DATASET%" --sync-labels
goto end

:export
.venv\Scripts\python.exe dataset_curator_yolo.py "%DATASET%" --export-folder
goto end

:analyse
.venv\Scripts\python.exe dataset_curator_yolo.py "%DATASET%"

:end
echo.
pause
