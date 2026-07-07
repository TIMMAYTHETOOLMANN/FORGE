@echo off
setlocal EnableDelayedExpansion
title NEXUS GGUF - Full System Deployment
set "ROOT=%~dp0"
set "VENV_PY=%ROOT%.venv\Scripts\python.exe"

REM Auto-detect Node.js in default installation directories
where node >nul 2>&1
if errorlevel 1 (
    if exist "C:\Program Files\nodejs\node.exe" (
        set "PATH=C:\Program Files\nodejs;!PATH!"
    ) else if exist "C:\Program Files (x86)\nodejs\node.exe" (
        set "PATH=C:\Program Files\nodejs;!PATH!"
    )
)

REM Auto-detect Python in default installation directories
where python >nul 2>&1
if errorlevel 1 (
    for /d %%p in ("C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python*") do (
        if exist "%%p\python.exe" (
            set "PATH=%%p;!PATH!"
        )
    )
    for /d %%p in ("C:\Program Files\Python*") do (
        if exist "%%p\python.exe" (
            set "PATH=%%p;!PATH!"
        )
    )
)

echo ==================================================
echo        NEXUS GGUF - FULL SYSTEM DEPLOYMENT
echo   Native Tuning Window  +  Web Forge Console
echo ==================================================
echo.

REM ---------- Preflight ----------
where node >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Node.js not found on PATH. Install Node.js 18+ and retry.
    pause & exit /b 1
)
if not exist "%VENV_PY%" (
    echo [ERROR] Python venv missing at "%VENV_PY%".
    echo         Recreate it and run: pip install -r requirements.txt
    pause & exit /b 1
)

REM ---------- 1. Native tuning window ----------
echo [1/2] Launching native tuning window (nexus_gguf_editor.py)...
start "NEXUS Editor Engine" cmd /k ""%VENV_PY%" "%ROOT%nexus_gguf_editor.py""

REM ---------- 2. Web console ----------
echo [2/2] Starting web console at http://localhost:3000 ...
cd /d "%ROOT%FRONTEND"

where pnpm >nul 2>&1
if errorlevel 1 (
    echo Installing pnpm...
    call npm install -g pnpm
)

if not exist "node_modules\next\dist\bin\next" (
    echo Installing frontend dependencies ^(first run^)...
    call pnpm install
)

REM Open the browser once Next has had a few seconds to boot (detached waiter).
start "" cmd /c "timeout /t 7 /nobreak >nul & start "" http://localhost:3000"

echo.
echo Web console booting. This window is the Next.js dev server log.
echo Close this window (or Ctrl+C) to stop the web console.
echo.
node "node_modules\next\dist\bin\next" dev
