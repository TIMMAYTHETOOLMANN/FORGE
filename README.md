# NEXUS GGUF Forge & Web Console

A comprehensive Next.js web environment and Python toolchain for forging, customizing, and serving GGUF models. 

This repository allows you to apply specialized neural surgical modifications (Persona directives, DeFi auditing, offensive security tuning, stealth operations, and custom tool injection) directly into your GGUF models, and deploy an Ollama-compatible backend server instantly.

---

## 🚀 Key Features

* **GGUF Neural Surgery Engine:** Inject system directives, persona layers, and toolsets directly into GGUF containers.
* **SL8 Custom Model Ranking:** Use the 8-lever UI slider system to tune persona priorities (defi_audit, offensive_security, exploit_engineering, stealth_ops, aggression, reasoning, creativity, loyalty).
* **Next.js Web Console:** A fully responsive, modern web interface running on port `3000` to stage, configure, and forge custom models.
* **Ollama-Compatible API Server:** Instant backend deployment serving your forged GGUF container on port `11435`.
* **Safe, Automated Toolchains:** Fully isolated script sequences for clean-room installations, environment repairs, and deployments.

---

## 🛠️ Prerequisites

* **Node.js:** v18.0 or newer
* **Python:** v3.10 or newer (with standard Tkinter support)
* **Package Manager:** `pnpm` (automatically bootstrapped)

---

## 🚀 Getting Started & Rebuilding Environment

If you have performed a system clean-up or your path environment has been corrupted, follow this systematic recovery sequence:

### 1. Fix System Environment PATH
If Node.js is installed but not accessible globally, run the path repair tool:
```cmd
FIX_SYSTEM_PATH.bat
```
*Note: Make sure to restart your command prompt or terminal after running this to apply permanent registry environment updates.*

### 2. Run Complete System Redeployment
Execute the clean build and validation script to verify python virtual environments, download missing dependencies, and prepare the Next.js server:
```cmd
REDEPLOY_FRESH.bat
```

---

## 🖥️ Launching the Forge

Once your environment is fully recovered, you can use any of the dedicated launch scripts:

* **`DEPLOY_FORGE.bat` (Recommended):** Systematically validates your environment, installs dependencies, boots the Next.js server, and auto-opens http://localhost:3000.
* **`LAUNCH_NEXUS_FULL.bat`:** Boots both the **Native Tkinter Tuning GUI** and the **Next.js Web Console** in parallel.
* **`LAUNCH_VENOM_NEXUS.bat`:** Boots only the Next.js Web Console dev server.
* **`DEPLOY\LAUNCH_SERVER.bat`:** Starts the compiled Flask/Llama-CPP model server on port `11435` to serve your staged/forged models.

---

## ⚙️ Repository Structure

* `FRONTEND/`: The Next.js 16 + React 19 web interface utilizing modern web standards.
* `DEPLOY/`: Compiled build targets, model serving assets, and shell scripts.
* `gguf_engine.py`: Core GGUF file read/write, persona surgery, and metadata injection module.
* `forge_toolchain.py`: Custom tool-selection core, GGUF curing, and local server routines.
* `venom_cli.py`: Streamlined CLI wrapper exposing forge operations.
* `forge_cfg_real.json`: Active JSON state holding your staged rankings, hardware targets, and persona configurations.
