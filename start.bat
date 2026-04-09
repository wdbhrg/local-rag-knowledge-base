@echo off
REM ASCII-only. Paths use batch-arg tilde-dp0 (folder of this script), not CD.
chcp 65001 >nul 2>&1
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"
if errorlevel 1 (
  echo [ERROR] Cannot cd to script folder.
  pause
  endlocal
  exit /b 1
)
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
title KB Launcher

REM Conda: override MINICONDA_ROOT if yours is elsewhere. Set KB_PYTHON to force venv instead of conda.
if not defined MINICONDA_ROOT set "MINICONDA_ROOT=E:\Project Tools\Miniconda3"
set "CONDA_EXE=%MINICONDA_ROOT%\Scripts\conda.exe"
set "USE_CONDA=0"
if exist "%CONDA_EXE%" if not defined KB_PYTHON set "USE_CONDA=1"

REM Script dir has trailing backslash; join as shown on the next lines.
if "!USE_CONDA!"=="1" (
  set "CONDA_PREFIX=%~dp0.conda_env"
  set "VENV_PY=%~dp0.conda_env\python.exe"
) else (
  set "CONDA_PREFIX="
  set "VENV_PY=%~dp0.venv\Scripts\python.exe"
)
set "REQ=%~dp0requirements.txt"
set "DB=%~dp0knowledge_base.sqlite"
set "BUILD_KB=%~dp0build_kb.py"
set "LAUNCH=%~dp0launch_ui.py"
set "MATERIALS_ROOT=%~dp0.."
set "ERR=0"
set "KB_DEBUG=0"

if "!USE_CONDA!"=="1" goto :after_have_any
set "HAVE_ANY=0"
where python >nul 2>&1 && set "HAVE_ANY=1"
if "!HAVE_ANY!"=="0" where py >nul 2>&1 && set "HAVE_ANY=1"
if "!HAVE_ANY!"=="0" (
  echo [ERROR] Neither python nor py on PATH. Install Python 3.10+ or fix MINICONDA_ROOT / conda.
  set "ERR=1"
  goto :hard_pause
)
:after_have_any

if /i "%~1"=="setup" (
  call :do_full_setup
  goto :hard_pause
)
if /i "%~1"=="setup-r1" (
  call :do_setup_r1
  goto :hard_pause
)
if /i "%~1"=="rebuild" (
  call :do_rebuild
  goto :hard_pause
)
if /i "%~1"=="shell" (
  call :do_shell
  endlocal
  exit /b 0
)
if /i "%~1"=="debug" set "KB_DEBUG=1"

call :do_launch_ui
if "!KB_DEBUG!"=="1" (
  echo.
  echo [DEBUG] Press any key to open a shell in this folder...
  pause >nul
  cmd /k
  endlocal
  exit /b 0
)
goto :hard_pause

:ensure_venv
if exist "%VENV_PY%" exit /b 0

if "!USE_CONDA!"=="1" (
  echo Creating conda env at "!CONDA_PREFIX!" ...
  if not exist "%CONDA_EXE%" (
    echo [ERROR] conda.exe not found: "%CONDA_EXE%"
    exit /b 1
  )
  "%CONDA_EXE%" create --prefix "!CONDA_PREFIX!" python=3.11 pip -y
  if not exist "%VENV_PY%" (
    echo [ERROR] conda create failed. Remove broken folder "!CONDA_PREFIX!" and retry.
    exit /b 1
  )
  exit /b 0
)

echo Creating .venv in %~dp0.venv ...

if defined KB_PYTHON (
  if not exist "%KB_PYTHON%" (
    echo [ERROR] KB_PYTHON set but file not found: "%KB_PYTHON%"
    exit /b 1
  )
  echo Using KB_PYTHON=%KB_PYTHON%
  "%KB_PYTHON%" -m venv "%~dp0.venv"
  if exist "%VENV_PY%" exit /b 0
  echo [ERROR] KB_PYTHON -m venv failed.
  exit /b 1
)

where python >nul 2>&1
if not errorlevel 1 (
  echo Trying: python -m venv ...
  python -m venv "%~dp0.venv"
  if exist "%VENV_PY%" exit /b 0
  echo [WARN] python -m venv failed or incomplete. Trying py -3 ...
)

where py >nul 2>&1
if not errorlevel 1 (
  echo Trying: py -3 -m venv ...
  py -3 -m venv "%~dp0.venv"
  if exist "%VENV_PY%" exit /b 0
)

echo [ERROR] Could not create .venv\Scripts\python.exe
echo [HINT] py -3 may point to a missing Python ^(e.g. C:\Python314\python.exe^). Use a working install.
echo [HINT] Install/repair from https://www.python.org/downloads/  ^(enable Add to PATH^).
echo [HINT] List py targets:  py -0p
echo [HINT] Or: set KB_PYTHON=C:\path\to\python.exe  then start.bat again
exit /b 1

:ensure_pip
if not exist "%REQ%" (
  echo [ERROR] File not found: "%REQ%"
  exit /b 1
)
if not exist "%VENV_PY%" (
  echo [ERROR] File not found: "%VENV_PY%"
  exit /b 1
)
echo Running: pip install -r "%REQ%"
"%VENV_PY%" -m pip install -r "%REQ%"
if errorlevel 1 (
  echo [ERROR] pip install failed.
  exit /b 1
)
exit /b 0

:do_launch_ui
call :ensure_venv
if errorlevel 1 exit /b 1
call :ensure_pip
if errorlevel 1 exit /b 1
if not exist "%DB%" (
  echo First-time index build...
  "%VENV_PY%" "%BUILD_KB%" --root "%MATERIALS_ROOT%" --db "%DB%"
  if errorlevel 1 (
    echo [ERROR] build_kb.py failed.
    exit /b 1
  )
)
echo Starting Streamlit...
"%VENV_PY%" "%LAUNCH%"
if errorlevel 1 (
  echo.
  echo [HINT] Web UI exited with an error. Scroll up for details.
)
exit /b 0

:do_full_setup
set "ERR=0"
call :ensure_venv
if errorlevel 1 (set "ERR=1" & goto :eof)
call :ensure_pip
if errorlevel 1 (set "ERR=1" & goto :eof)
if not exist "%DB%" (
  echo [1/2] Building index...
  "%VENV_PY%" "%BUILD_KB%" --root "%MATERIALS_ROOT%" --db "%DB%"
  if errorlevel 1 (set "ERR=1" & goto :eof)
) else (
  echo [1/2] Index exists, skip.
)
set "OLLAMA_EXE=!LocalAppData!\Programs\Ollama\ollama.exe"
if not exist "!OLLAMA_EXE!" set "OLLAMA_EXE="
if not defined OLLAMA_EXE where ollama >nul 2>&1 && set "OLLAMA_EXE=ollama"
if not defined OLLAMA_EXE (
  echo [2/2] Ollama not found, skip models.
  goto :eof
)
echo [2/2] ollama pull qwen2.5:7b ^(may take a while^)...
pushd "%~dp0" >nul
"!OLLAMA_EXE!" pull qwen2.5:7b
set "PULL_ERR=!errorlevel!"
popd >nul
if not "!PULL_ERR!"=="0" (
  echo [ERROR] ollama pull failed.
  set "ERR=1"
  goto :eof
)
echo [2/2] ollama create kb-rag...
pushd "%~dp0" >nul
"!OLLAMA_EXE!" create kb-rag -f "%~dp0Modelfile"
set "CREATE_ERR=!errorlevel!"
popd >nul
if not "!CREATE_ERR!"=="0" (
  echo [ERROR] ollama create kb-rag failed.
  set "ERR=1"
  goto :eof
)
echo.
echo Done. In UI sidebar choose local model kb-rag (or refresh detection).
goto :eof

:do_setup_r1
set "ERR=0"
set "OLLAMA_EXE=!LocalAppData!\Programs\Ollama\ollama.exe"
if not exist "!OLLAMA_EXE!" set "OLLAMA_EXE="
if not defined OLLAMA_EXE where ollama >nul 2>&1 && set "OLLAMA_EXE=ollama"
if not defined OLLAMA_EXE (
  echo [ERROR] Ollama not found.
  set "ERR=1"
  goto :eof
)
echo [1/2] ollama pull deepseek-r1:8b...
pushd "%~dp0" >nul
"!OLLAMA_EXE!" pull deepseek-r1:8b
if errorlevel 1 (
  popd >nul
  echo [ERROR] ollama pull failed.
  set "ERR=1"
  goto :eof
)
echo [2/2] ollama create kb-rag-r1-8b...
"!OLLAMA_EXE!" create kb-rag-r1-8b -f "%~dp0Modelfile.deepseek-r1"
set "R1_ERR=!errorlevel!"
popd >nul
if not "!R1_ERR!"=="0" (
  echo [ERROR] ollama create failed.
  set "ERR=1"
  goto :eof
)
echo.
echo Done. In UI choose model kb-rag-r1-8b.
goto :eof

:do_rebuild
call :ensure_venv
if errorlevel 1 exit /b 1
call :ensure_pip
if errorlevel 1 exit /b 1
echo Rebuilding knowledge_base.sqlite...
"%VENV_PY%" "%BUILD_KB%" --root "%MATERIALS_ROOT%" --db "%DB%"
if errorlevel 1 (echo [ERROR] rebuild failed.) else (echo Rebuild OK.)
exit /b 0

:do_shell
call :ensure_venv
if errorlevel 1 goto :hard_pause
call :ensure_pip
if errorlevel 1 goto :hard_pause
cd /d "%~dp0"
cmd /k
exit /b 0

:hard_pause
echo.
if "!ERR!"=="1" echo [HINT] From cmd: cd /d "%~dp0" ^& "%~nx0"
pause
endlocal
exit /b 0
