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
set "CUDA_CUBLAS=ON"

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
if /I "%~1"=="--cuda-reference" (
    set "CUDA=ON"
    set "CUDA_CUBLAS=OFF"
    shift
    goto parse_args
)
if /I "%~1"=="-h" goto usage_ok
if /I "%~1"=="--help" goto usage_ok
echo Unknown option: %~1
goto usage_error

:configure
if "%CLEAN%"=="1" if exist "%BUILD_DIR%" rmdir /S /Q "%BUILD_DIR%"

cmake -S "%ROOT_DIR%" -B "%BUILD_DIR%" -DCITRIUS_ENABLE_METAL=OFF -DCITRIUS_ENABLE_CUDA=%CUDA% -DCITRIUS_CUDA_USE_CUBLAS=%CUDA_CUBLAS%
if errorlevel 1 exit /B %errorlevel%

cmake --build "%BUILD_DIR%" -j --config "%CONFIG%"
exit /B %errorlevel%

:missing_config
echo --config requires a value, such as Debug or Release.
goto usage_error

:usage_ok
echo Usage: build.bat [options]
echo.
echo Options:
echo   --cuda              Enable CUDA and use cuBLAS as the default top-level
echo                       CUDA matmul backend.
echo   --cuda-reference    Enable CUDA and use the Citrius reference CUDA kernel
echo                       as the default top-level matmul backend.
echo   --clean             Remove the build directory before configuring.
echo   --config CONFIG     Build Debug, Release, RelWithDebInfo, or MinSizeRel.
echo   -h, --help          Show this help.
echo.
echo With neither CUDA option, CUDA is disabled. The corresponding CMake values are:
echo.
echo   build.bat                    CITRIUS_ENABLE_CUDA=OFF
echo   build.bat --cuda             CITRIUS_ENABLE_CUDA=ON
echo                                CITRIUS_CUDA_USE_CUBLAS=ON
echo   build.bat --cuda-reference   CITRIUS_ENABLE_CUDA=ON
echo                                CITRIUS_CUDA_USE_CUBLAS=OFF
echo.
echo CITRIUS_CUDA_USE_CUBLAS only selects the compiled default for top-level CUDA
echo matmul. It does not enable or disable CUDA. At runtime, CITRIUS_CUDA_BACKEND
echo can override the default with cublas, cutlass, or reference.
exit /B 0

:usage_error
echo Usage: build.bat [--clean] [--cuda^|--cuda-reference] [--config Debug^|Release^|RelWithDebInfo^|MinSizeRel]
exit /B 1
