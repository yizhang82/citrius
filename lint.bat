@echo off
setlocal EnableExtensions

for %%I in ("%~dp0.") do set "ROOT_DIR=%%~fI"
set "BUILD_DIR=%ROOT_DIR%\build"

if /I "%~1"=="-h" goto usage
if /I "%~1"=="--help" goto usage
if not "%~1"=="" goto usage_error

if not exist "%BUILD_DIR%\CMakeCache.txt" (
    echo Build directory is not configured. Run build.bat first.
    exit /B 1
)

cmake --build "%BUILD_DIR%" --target format
exit /B %errorlevel%

:usage
echo Usage: lint.bat
echo Formats Citrius C++ and CUDA sources using the clang-format found by CMake.
exit /B 0

:usage_error
echo Unknown option: %~1
echo Usage: lint.bat
exit /B 1
