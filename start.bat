@echo off
REM ===================================================================
REM  IndexTTS-2.5 Pro  -  launcher
REM
REM  Checks the environment (venv / dependencies / model files / CUDA),
REM  offers to repair what is missing, then starts the Pro console and
REM  opens the WebUI in your browser.
REM
REM  Extra arguments are passed through, e.g.:
REM      start.bat --lazy              do not preload the model
REM      start.bat --port 7861         use another port
REM      start.bat --host 0.0.0.0      expose on the LAN
REM
REM  NOTE: this file is deliberately ASCII-only. cmd.exe parses batch
REM  files using the *console code page*, so non-ASCII text here is
REM  mis-decoded on machines whose code page differs from the author's
REM  and silently corrupts the surrounding commands. Keep it ASCII.
REM ===================================================================

setlocal EnableExtensions
cd /d "%~dp0"
title IndexTTS-2.5 Pro

set "PY=.venv\Scripts\python.exe"

echo ============================================================
echo   IndexTTS-2.5 Pro  -  environment check
echo ============================================================
echo.

REM ---------------------------------------------------------------- 1) venv
if not exist "%PY%" goto :no_venv
"%PY%" -c "import sys" >nul 2>&1
if errorlevel 1 goto :bad_venv
echo   [OK] venv            %PY%

REM -------------------------------------------------------- 2) dependencies
set "MISS="
for /f "usebackq delims=" %%m in (`"%PY%" -c "import importlib.util as u;print(' '.join([m for m in ('torch','torchaudio','gradio','peft','pypinyin','soundfile','librosa','transformers','omegaconf','whisper','numpy','tiktoken') if not u.find_spec(m)]))" 2^>nul`) do set "MISS=%%m"
if not "%MISS%"=="" goto :miss_deps

REM  NOTE: no '%' characters in these one-liners - cmd.exe would expand
REM  '%s' as an (undefined) environment variable and silently blank it.
"%PY%" -c "import torch;print('   [OK] pytorch        ', torch.__version__, '/ cuda', torch.version.cuda)" 2>nul
"%PY%" -c "import torch;print('   [OK] gpu            ', (torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no CUDA - falling back to CPU (slow)'))" 2>nul

REM -------------------------------------------------------------- 3) models
set "MISS_MDL="
for /f "usebackq delims=" %%m in (`"%PY%" -c "import os;need=['gpt.pth','codec.pth','s2mel.pth','config.yaml','multilingual_zh_ja_yue_char_del.tiktoken','wav2vec2bert_stats.pt','feat1.pt','feat2.pt']+['hf_cache/w2v-bert-2.0','hf_cache/campplus_cn_common.bin','hf_cache/semantic_codec','hf_cache/bigvgan'];print(' '.join([f for f in need if not os.path.exists(os.path.join('checkpoints',f))]))" 2^>nul`) do set "MISS_MDL=%%m"
if not "%MISS_MDL%"=="" goto :miss_models
echo   [OK] models          main + auxiliary checkpoints all present

echo.
echo ============================================================
echo   Ready. Starting the WebUI ...
echo   The browser will open at http://127.0.0.1:7860
echo   Close this window (or press Ctrl+C) to stop the server.
echo ============================================================
echo.

"%PY%" webui_pro.py %*
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" (
    echo.
    echo [X] WebUI exited with code %RC%
    echo     If the port is already in use try:  start.bat --port 7861
    echo.
    pause
)
endlocal
exit /b %RC%


REM ===================================================================
REM  Repair / error branches
REM ===================================================================
:no_venv
echo   [X] virtual environment not found: %PY%
echo.
echo       This project manages its environment with uv.
echo       From the project root, run:
echo.
echo           uv sync --extra webui --default-index "https://mirrors.aliyun.com/pypi/simple"
echo.
echo       --extra webui adds gradio on top of the base dependencies.
echo       Do NOT use --all-extras: it also pulls deepspeed / flash-attn /
echo       torch_compile, which need a CUDA toolchain and are optional here.
echo.
echo       First run downloads roughly 5-8 GB (Python + torch/CUDA + deps).
echo       When it finishes, double-click this file again.
echo.
echo       uv itself not installed yet? Run:  pip install uv
echo.
pause
endlocal
exit /b 1

:bad_venv
echo   [X] the venv exists but cannot run: %PY%
echo       It may be corrupt. Delete the .venv folder, then run:
echo           uv sync --extra webui
echo.
pause
endlocal
exit /b 1

:miss_deps
echo   [!] missing dependencies: %MISS%
echo.
echo       Reason: the upstream pyproject.toml does not declare everything
echo       this fork needs. `peft` powers LoRA training, `pypinyin` powers
echo       pronunciation correction, and `gradio` lives in --extra webui,
echo       so a plain `uv sync` leaves them out.
echo.
where uv >nul 2>&1
if errorlevel 1 goto :no_uv
set "ANS="
set /p "ANS=      Install the missing packages with uv now? [Y/N] "
if /i not "%ANS%"=="Y" goto :deps_skip
echo.
echo       Installing: %MISS%
uv pip install --python "%PY%" %MISS% --default-index "https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple"
if errorlevel 1 goto :deps_fail
echo.
echo       Done. Re-checking ...
echo.
endlocal
call "%~f0" %*
exit /b %ERRORLEVEL%

:no_uv
echo       `uv` was not found on PATH. Either install it:  pip install uv
echo       or install the packages manually:
echo           .venv\Scripts\python.exe -m pip install %MISS%
echo.
pause
endlocal
exit /b 1

:deps_skip
echo.
echo       Skipped. To install manually:
echo           uv pip install --python .venv\Scripts\python.exe %MISS%
echo.
pause
endlocal
exit /b 1

:deps_fail
echo.
echo   [X] install failed. Check your network and retry, or run the command above.
echo.
pause
endlocal
exit /b 1

:miss_models
echo   [!] checkpoints is missing: %MISS_MDL%
echo.
echo       Download them first from the "Models" tab of the WebUI, or run:
echo           .venv\Scripts\python.exe tools\model_fetcher.py --version 2.5 --all
echo.
set "ANS="
set /p "ANS=      Start anyway? Tabs that need the missing files will error. [Y/N] "
if /i not "%ANS%"=="Y" goto :models_cancel
echo.
echo       Starting with an incomplete model set ...
echo.
"%PY%" webui_pro.py %*
endlocal
exit /b %ERRORLEVEL%

:models_cancel
echo.
echo       Cancelled. Re-run this file once the models are in place.
echo.
pause
endlocal
exit /b 1
