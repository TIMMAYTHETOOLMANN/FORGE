@echo off
setlocal EnableDelayedExpansion
title VENOM Nexus - Neural Forging Interface

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

cd /d "%~dp0FRONTEND"

echo ============================================
echo        VENOM Nexus - Neural Forging UI
echo             Next.js web console
echo ============================================
echo.

where node >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Node.js not found on PATH. Install Node.js 18+ and retry.
    pause
    exit /b 1
)

where pnpm >nul 2>&1
if errorlevel 1 (
    echo Installing pnpm...
    call npm install -g pnpm
)

if not exist "node_modules\next\dist\bin\next" (
    echo Installing frontend dependencies ^(first run^)...
    call pnpm install
)

echo Starting VENOM Nexus on http://localhost:3000 ...
:: Bypass the pnpm run-script wrapper (its verify-deps preflight aborts on the
:: optional sharp build script) and invoke the Next binary directly.
start "" "http://localhost:3000"
node "node_modules\next\dist\bin\next" dev
