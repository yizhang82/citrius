@echo off
setlocal EnableExtensions

for %%I in ("%~dp0.") do set "ROOT_DIR=%%~fI"
set "BUILD_DIR=%ROOT_DIR%\build"
set "BACKEND="
set "CUDA_ARG="

if /I "%~1"=="--cpu" set "BACKEND=--cpu"
if /I "%~1"=="--cuda" (
    set "BACKEND=--cuda"
    set "CUDA_ARG=--cuda"
)
if /I "%~1"=="--all" (
    set "BACKEND=--all"
    set "CUDA_ARG=--cuda"
)
if not "%~2"=="" goto usage_error
if not defined BACKEND goto usage_error

call "%ROOT_DIR%\build.bat" %CUDA_ARG% --config Release
if errorlevel 1 exit /B %errorlevel%

"%BUILD_DIR%\Release\operations_benchmark.exe" %BACKEND%
exit /B %errorlevel%

:usage_error
echo Usage: benchmark.bat --cpu^|--cuda^|--all
exit /B 1
