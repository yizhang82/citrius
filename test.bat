@echo off
setlocal EnableExtensions

for %%I in ("%~dp0.") do set "ROOT_DIR=%%~fI"
set "BUILD_DIR=%ROOT_DIR%\build"
set "CONFIG=Release"
set "CLEAN_ARG="

:parse_args
if "%~1"=="" goto build
if /I "%~1"=="--clean" (
    set "CLEAN_ARG=--clean"
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
if /I "%~1"=="-h" goto usage_ok
if /I "%~1"=="--help" goto usage_ok
echo Unknown option: %~1
goto usage_error

:build
call "%ROOT_DIR%\build.bat" %CLEAN_ARG% --config "%CONFIG%"
if errorlevel 1 exit /B %errorlevel%

ctest --test-dir "%BUILD_DIR%" --build-config "%CONFIG%" --output-on-failure
exit /B %errorlevel%

:missing_config
echo --config requires a value, such as Debug or Release.
goto usage_error

:usage_ok
echo Usage: test.bat [--clean] [--config Debug^|Release^|RelWithDebInfo^|MinSizeRel]
exit /B 0

:usage_error
echo Usage: test.bat [--clean] [--config Debug^|Release^|RelWithDebInfo^|MinSizeRel]
exit /B 1
