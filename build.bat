@echo off
setlocal EnableExtensions

rem MSBuild treats Path and PATH as duplicate keys if both are inherited.
set "CITRIUS_SAVED_PATH=%PATH%"
set "PATH="
set "Path=%CITRIUS_SAVED_PATH%"
set "CITRIUS_SAVED_PATH="

for %%I in ("%~dp0.") do set "ROOT_DIR=%%~fI"
set "BUILD_DIR=%ROOT_DIR%\build"
set "CONFIG=Release"
set "CLEAN=0"
set "CUDA=OFF"

:parse_args
if "%~1"=="" goto configure
if /I "%~1"=="--clean" (
    set "CLEAN=1"
    shift
    goto parse_args
)
if /I "%~1"=="--config" (
    if "%~2"=="" goto missing_config
    set "CONFIG=%~2"
    shift
    shift
    goto parse_args
)
if /I "%~1"=="--cuda" (
    set "CUDA=ON"
    shift
    goto parse_args
)
if /I "%~1"=="-h" goto usage_ok
if /I "%~1"=="--help" goto usage_ok
echo Unknown option: %~1
goto usage_error

:configure
if "%CLEAN%"=="1" if exist "%BUILD_DIR%" rmdir /S /Q "%BUILD_DIR%"

cmake -S "%ROOT_DIR%" -B "%BUILD_DIR%" -DCITRIUS_ENABLE_METAL=OFF -DCITRIUS_ENABLE_CUDA=%CUDA%
if errorlevel 1 exit /B %errorlevel%

cmake --build "%BUILD_DIR%" --config "%CONFIG%"
exit /B %errorlevel%

:missing_config
echo --config requires a value, such as Debug or Release.
goto usage_error

:usage_ok
echo Usage: build.bat [--clean] [--cuda] [--config Debug^|Release^|RelWithDebInfo^|MinSizeRel]
exit /B 0

:usage_error
echo Usage: build.bat [--clean] [--cuda] [--config Debug^|Release^|RelWithDebInfo^|MinSizeRel]
exit /B 1
