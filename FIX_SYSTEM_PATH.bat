@echo off
setlocal EnableDelayedExpansion
title NEXUS GGUF - Environment PATH Repair
echo.
echo ================================================================
echo   NEXUS GGUF - AUTOMATIC SYSTEM PATH REPAIR
echo ================================================================
echo.

REM Check if Node is already accessible
where node >nul 2>&1
if %errorlevel% equ 0 (
    echo [INFO] Node.js is already configured and accessible in your PATH!
    echo Version: 
    node --version
    echo.
    pause
    exit /b 0
)

REM Check default installation locations
set "NODE_DIR=C:\Program Files\nodejs"
if not exist "!NODE_DIR!\node.exe" (
    set "NODE_DIR=C:\Program Files (x86)\nodejs"
)

if not exist "!NODE_DIR!\node.exe" (
    echo [ERROR] Node.js installation not found in default locations.
    echo Please download and install Node.js from: https://nodejs.org/
    echo.
    pause
    exit /b 1
)

echo [FOUND] Node.js detected at: "!NODE_DIR!"
echo.
echo Attempting to permanently add Node.js to your User PATH...

REM Retrieve current User PATH
set "USER_PATH="
for /f "usebackq tokens=2*" %%a in (`reg query HKCU\Environment /v PATH 2^>nul`) do (
    set "USER_PATH=%%b"
)

REM Check if already present in USER_PATH string
if defined USER_PATH (
    echo !USER_PATH! | find /i "!NODE_DIR!" >nul
    if !errorlevel! equ 0 (
        echo [INFO] Node.js directory is already in your registry User PATH.
        echo You may just need to restart your terminal or computer.
        goto DONE
    )
    set "NEW_PATH=!USER_PATH!;!NODE_DIR!"
) else (
    set "NEW_PATH=!NODE_DIR!"
)

REM Update User Environment Registry
reg add HKCU\Environment /v PATH /t REG_EXPAND_SZ /d "!NEW_PATH!" /f >nul
if !errorlevel! equ 0 (
    echo [SUCCESS] Node.js was permanently added to your User PATH!
    echo.
    echo [IMPORTANT] Please CLOSE this terminal and open a NEW Command Prompt
    echo             for the changes to take effect.
) else (
    echo [ERROR] Failed to update registry. Attempting setx fallback...
    setx PATH "!PATH!;!NODE_DIR!"
)

:DONE
echo.
pause
