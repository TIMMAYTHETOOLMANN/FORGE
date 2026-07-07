@echo off
setlocal EnableDelayedExpansion
title NEXUS GGUF - Complete System Redeployment
set "ROOT=%~dp0"
set "VENV_PY=%ROOT%.venv\Scripts\python.exe"
set "DEPLOY_DIR=%ROOT%DEPLOY"
set "FRONTEND_DIR=%ROOT%FRONTEND"

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
echo ================================================================
echo   NEXUS GGUF - COMPLETE SYSTEM REDEPLOYMENT (Fresh Config)
echo ================================================================
echo.
echo [Phase 1/5] Cleaning previous deployment state...
echo ----------------------------------------------------------------

REM Clear old GGUF deployments in DEPLOY folder
if exist "%DEPLOY_DIR%\*.gguf" (
    echo   Removing old GGUF files from DEPLOY...
    del /q "%DEPLOY_DIR%\*.gguf" 2>nul
)

REM Clear __pycache__ directories
for /d /r "%ROOT%" %%d in (__pycache__) do (
    if exist "%%d" (
        echo   Clearing cache: %%d
        rd /s /q "%%d" 2>nul
    )
)

REM Clear Python bytecode
del /s /q "%ROOT%*.pyc" 2>nul

echo   Done.
echo.

echo [Phase 2/5] Resetting forge configuration...
echo ----------------------------------------------------------------
echo   Applying fresh GGUF config from forge_cfg_fresh.json...
copy /y "%ROOT%forge_cfg_fresh.json" "%ROOT%forge_cfg_real.json" >nul
echo   Config reset complete.
echo.

echo [Phase 3/5] Validating Python backend environment...
echo ----------------------------------------------------------------
if not exist "%VENV_PY%" (
    echo   [!] Virtual environment not found. Creating fresh venv...
    python -m venv "%ROOT%.venv"
    if errorlevel 1 (
        echo   [ERROR] Failed to create venv. Ensure Python 3.10+ is installed.
        pause
        exit /b 1
    )
)

echo   Installing/updating backend dependencies...
"%VENV_PY%" -m pip install --upgrade pip -q
"%VENV_PY%" -m pip install -r "%ROOT%requirements.txt" -q
if errorlevel 1 (
    echo   [WARN] Some packages may have failed. Continuing...
) else (
    echo   Dependencies installed successfully.
)
echo.

echo [Phase 4/5] Preparing Next.js frontend...
echo ----------------------------------------------------------------
cd /d "%FRONTEND_DIR%"

where node >nul 2>&1
if errorlevel 1 (
    echo   [ERROR] Node.js not found on PATH. Install Node.js 18+ and retry.
    pause
    exit /b 1
)

where pnpm >nul 2>&1
if errorlevel 1 (
    echo   Installing pnpm globally...
    npm install -g pnpm
)

if exist "node_modules" (
    echo   Cleaning existing node_modules...
    rd /s /q "node_modules" 2>nul
)
if exist ".next" (
    echo   Cleaning .next build cache...
    rd /s /q ".next" 2>nul
)

echo   Installing fresh frontend dependencies...
call pnpm install
if errorlevel 1 (
    echo   [ERROR] pnpm install failed.
    pause
    exit /b 1
)
echo   Frontend ready.
echo.

cd /d "%ROOT%"

echo [Phase 5/5] System redeployment complete!
echo ----------------------------------------------------------------
echo.
echo   Configuration: forge_cfg_real.json (reset to fresh defaults)
echo   Backend:       Python venv ready
echo   Frontend:      Next.js dependencies installed
echo   Deployment:    Port 11435 (backend), Port 3000 (frontend)
echo.
echo ================================================================
echo   Ready to launch. Use one of:
echo     - LAUNCH_NEXUS_FULL.bat    (Editor + Web Console)
echo     - LAUNCH_VENOM_NEXUS.bat   (Web Console only)
echo     - DEPLOY\LAUNCH_SERVER.bat (API server only)
echo ================================================================
echo.

echo.
echo Starting NEXUS GGUF Full System (Autonomous Auto-Launch)...
echo.
call "%ROOT%LAUNCH_NEXUS_FULL.bat"
