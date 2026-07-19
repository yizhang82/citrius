@echo off
setlocal EnableExtensions

for %%I in ("%~dp0.") do set "ROOT_DIR=%%~fI"
set "BUILD_DIR=%ROOT_DIR%\build"
set "TEST=%~1"
set "BACKEND="
set "CUDA_ARG="
set "BENCH_ARGS="
set "REPORT_ARGS="
set "REPORT_PATH="

if not exist "%BUILD_DIR%\CMakeCache.txt" (
    echo Build directory is not configured. Run build.bat first.
    exit /B 1
)

if /I "%TEST%"=="operations" goto parse_operations
if /I "%TEST%"=="add-kernel" goto parse_add_kernel
if /I "%TEST%"=="matmul-kernel" goto parse_matmul_kernel
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
if not defined BACKEND goto usage_error

if "%~3"=="" goto run_operations
if /I not "%~3"=="--html" goto usage_error
set "REPORT_ARGS=--html"
if not "%~4"=="" set "REPORT_PATH=%~4"
if not "%~5"=="" goto usage_error

:run_operations

if defined CUDA_ARG call :require_cuda
if errorlevel 1 exit /B %errorlevel%

cmake --build "%BUILD_DIR%" --config Release --target operations_benchmark
if errorlevel 1 exit /B %errorlevel%

if defined REPORT_PATH (
    "%BUILD_DIR%\Release\operations_benchmark.exe" %BACKEND% --html "%REPORT_PATH%"
) else if defined REPORT_ARGS (
    "%BUILD_DIR%\Release\operations_benchmark.exe" %BACKEND% --html
) else (
    "%BUILD_DIR%\Release\operations_benchmark.exe" %BACKEND%
)
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
call :require_cuda
if errorlevel 1 exit /B %errorlevel%

cmake --build "%BUILD_DIR%" -j --config Release --target cuda_elementwise_benchmark
if errorlevel 1 exit /B %errorlevel%

"%BUILD_DIR%\Release\cuda_elementwise_benchmark.exe" %BENCH_ARGS%
exit /B %errorlevel%

:parse_matmul_kernel
shift

:parse_matmul_kernel_args
if "%~1"=="" goto run_matmul_kernel
if /I "%~1"=="--size" goto matmul_kernel_value
if /I "%~1"=="--iterations" goto matmul_kernel_value
if /I "%~1"=="--samples" goto matmul_kernel_value
goto usage_error

:matmul_kernel_value
if "%~2"=="" goto usage_error
set "BENCH_ARGS=%BENCH_ARGS% %~1 %~2"
shift
shift
goto parse_matmul_kernel_args

:run_matmul_kernel
call :require_cuda
if errorlevel 1 exit /B %errorlevel%

cmake --build "%BUILD_DIR%" -j --config Release --target cuda_matmul_benchmark
if errorlevel 1 exit /B %errorlevel%

"%BUILD_DIR%\Release\cuda_matmul_benchmark.exe" %BENCH_ARGS%
exit /B %errorlevel%

:require_cuda
findstr /X /C:"CITRIUS_ENABLE_CUDA:BOOL=ON" "%BUILD_DIR%\CMakeCache.txt" >nul
if errorlevel 1 (
    echo Existing build is not configured with CUDA. Run build.bat --cuda first.
    exit /B 1
)
exit /B 0

:usage_error
echo Usage:
echo   benchmark.bat operations --cpu^|--cuda^|--all [--html [FILE]]
echo   benchmark.bat add-kernel [--size N] [--iterations N] [--samples N]
echo   benchmark.bat matmul-kernel [--size N] [--iterations N] [--samples N]
exit /B 1
