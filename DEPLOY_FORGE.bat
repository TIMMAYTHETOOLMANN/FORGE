@echo off
setlocal EnableDelayedExpansion
title NEXUS GGUF Forge — Full Deployment
set "ROOT=%~dp0"
set "VENV_PY=%ROOT%.venv\Scripts\python.exe"
set "FRONTEND=%ROOT%FRONTEND"

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

echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║       NEXUS GGUF FORGE — FULL DEPLOYMENT SYSTEM              ║
echo ║   Configure custom models via web interface at :3000         ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.

REM ── Phase 1: Environment Check ──
echo [1/4] Checking environment...

where node >nul 2>&1
if errorlevel 1 (
    echo   [ERROR] Node.js not found. Install Node.js 18+ first.
    pause
    exit /b 1
)

if not exist "%VENV_PY%" (
    echo   [WARN] Python venv not found. Creating...
    python -m venv "%ROOT%.venv"
    if errorlevel 1 (
        echo   [ERROR] Failed to create venv.
        pause
        exit /b 1
    )
)

echo   ✓ Node.js found
echo   ✓ Python venv ready
echo.

REM ── Phase 2: Backend Dependencies ──
echo [2/4] Installing backend dependencies...
"%VENV_PY%" -m pip install --upgrade pip -q 2>nul
"%VENV_PY%" -m pip install -r "%ROOT%requirements.txt" -q 2>nul
echo   ✓ Python packages ready
echo.

REM ── Phase 3: Frontend Dependencies ──
echo [3/4] Preparing frontend...
cd /d "%FRONTEND%"

where pnpm >nul 2>&1
if errorlevel 1 (
    echo   Installing pnpm...
    npm install -g pnpm
)

if not exist "node_modules\next\dist\bin\next" (
    echo   Installing frontend dependencies (first run)...
    call pnpm install
)
echo   ✓ Frontend dependencies ready
echo.

REM ── Phase 4: Launch ──
echo [4/4] Launching Forge System...
echo.
echo ┌──────────────────────────────────────────────────────────────┐
echo │  FORGE ENDPOINTS:                                           │
echo │                                                              │
echo │  Web Forge UI:     http://localhost:3000                    │
echo │  Model API:        http://localhost:3000/api/models         │
echo │  Config API:       http://localhost:3000/api/config         │
echo │  Forge API:        http://localhost:3000/api/forge          │
echo │  Browse API:       http://localhost:3000/api/browse         │
echo │  Scan API:         http://localhost:3000/api/scan           │
echo │                                                              │
echo │  After forging, the model server runs on :11435             │
echo └──────────────────────────────────────────────────────────────┘
echo.

REM Open browser after brief delay
start "" cmd /c "timeout /t 4 /nobreak >nul & start "" http://localhost:3000"

echo Starting Next.js dev server...
echo Press Ctrl+C to stop.
echo.

node "node_modules\next\dist\bin\next" dev
