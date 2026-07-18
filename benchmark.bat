@echo off
setlocal EnableExtensions

for %%I in ("%~dp0.") do set "ROOT_DIR=%%~fI"
set "BUILD_DIR=%ROOT_DIR%\build"
set "TEST=%~1"
set "BACKEND="
set "CUDA_ARG="
set "BENCH_ARGS="

if /I "%TEST%"=="operations" goto parse_operations
if /I "%TEST%"=="add-kernel" goto parse_add_kernel
goto usage_error

:parse_operations
if /I "%~2"=="--cpu" set "BACKEND=--cpu"
if /I "%~2"=="--cuda" (
    set "BACKEND=--cuda"
    set "CUDA_ARG=--cuda"
)
if /I "%~2"=="--all" (
    set "BACKEND=--all"
    set "CUDA_ARG=--cuda"
)
if not "%~3"=="" goto usage_error
if not defined BACKEND goto usage_error

call "%ROOT_DIR%\build.bat" %CUDA_ARG% --config Release
if errorlevel 1 exit /B %errorlevel%

"%BUILD_DIR%\Release\operations_benchmark.exe" %BACKEND%
exit /B %errorlevel%

:parse_add_kernel
shift

:parse_add_kernel_args
if "%~1"=="" goto run_add_kernel
if /I "%~1"=="--size" goto add_kernel_value
if /I "%~1"=="--iterations" goto add_kernel_value
if /I "%~1"=="--samples" goto add_kernel_value
goto usage_error

:add_kernel_value
if "%~2"=="" goto usage_error
set "BENCH_ARGS=%BENCH_ARGS% %~1 %~2"
shift
shift
goto parse_add_kernel_args

:run_add_kernel
call "%ROOT_DIR%\build.bat" --cuda --config Release
if errorlevel 1 exit /B %errorlevel%

"%BUILD_DIR%\Release\cuda_elementwise_benchmark.exe" %BENCH_ARGS%
exit /B %errorlevel%

:usage_error
echo Usage:
echo   benchmark.bat operations --cpu^|--cuda^|--all
echo   benchmark.bat add-kernel [--size N] [--iterations N] [--samples N]
exit /B 1
