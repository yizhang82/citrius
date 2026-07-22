#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$ROOT_DIR/build"

usage() {
    echo "Usage:"
    echo "  ./benchmark.sh operations --cpu|--metal|--cuda|--all [--html [FILE]]"
    echo "  ./benchmark.sh qwen3-decoding --cpu|--cuda [--tokens N] [--dtype float32|float16|bfloat16]"
    echo "  ./benchmark.sh add-kernel [--size N] [--iterations N] [--samples N]"
    echo "  ./benchmark.sh matmul-kernel [--size N] [--iterations N] [--samples N]"
}

require_backend() {
    local backend="$1"
    local option="$2"
    local build_flag="$3"
    if ! grep -q "^${option}:BOOL=ON$" "$BUILD_DIR/CMakeCache.txt"; then
        echo "Existing build is not configured with $backend. Run ./build.sh $build_flag first."
        exit 1
    fi
}

if [[ ! -f "$BUILD_DIR/CMakeCache.txt" ]]; then
    echo "Build directory is not configured. Run ./build.sh first."
    exit 1
fi

[[ $# -gt 0 ]] || { usage; exit 1; }
benchmark="$1"
shift

case "$benchmark" in
    operations)
        [[ $# -gt 0 ]] || { usage; exit 1; }
        backend="$1"
        shift
        case "$backend" in
            --cpu) ;;
            --metal) require_backend "Metal" CITRIUS_ENABLE_METAL --metal ;;
            --cuda) require_backend "CUDA" CITRIUS_ENABLE_CUDA --cuda ;;
            --all)
                require_backend "CUDA" CITRIUS_ENABLE_CUDA --cuda
                ;;
            *) usage; exit 1 ;;
        esac

        report_args=()
        report_arg_count=0
        if [[ $# -gt 0 ]]; then
            [[ "$1" == "--html" ]] || { usage; exit 1; }
            report_args+=(--html)
            report_arg_count=1
            shift
            if [[ $# -gt 0 ]]; then
                report_args+=("$1")
                report_arg_count=2
                shift
            fi
        fi
        [[ $# -eq 0 ]] || { usage; exit 1; }

        cmake --build "$BUILD_DIR" -j --target operations_benchmark
        if [[ "$report_arg_count" -gt 0 ]]; then
            "$BUILD_DIR/operations_benchmark" "$backend" "${report_args[@]}"
        else
            "$BUILD_DIR/operations_benchmark" "$backend"
        fi
        ;;
    qwen3-decoding)
        [[ $# -ge 1 ]] || { usage; exit 1; }
        backend="$1"
        shift
        case "$backend" in
            --cpu) ;;
            --cuda) require_backend "CUDA" CITRIUS_ENABLE_CUDA --cuda ;;
            *) usage; exit 1 ;;
        esac
        benchmark_args=()
        while [[ $# -gt 0 ]]; do
            [[ ("$1" == "--tokens" || "$1" == "--dtype") && $# -ge 2 ]] || { usage; exit 1; }
            benchmark_args+=("$1" "$2")
            shift 2
        done
        cmake --build "$BUILD_DIR" -j --target qwen3_decoding_benchmark
        "$BUILD_DIR/qwen3_decoding_benchmark" "$backend" "${benchmark_args[@]}"
        ;;
    add-kernel|matmul-kernel)
        require_backend "CUDA" CITRIUS_ENABLE_CUDA --cuda
        benchmark_args=()
        benchmark_arg_count=0
        while [[ $# -gt 0 ]]; do
            case "$1" in
                --size|--iterations|--samples)
                    [[ $# -ge 2 ]] || { usage; exit 1; }
                    benchmark_args+=("$1" "$2")
                    benchmark_arg_count=$((benchmark_arg_count + 2))
                    shift 2
                    ;;
                *) usage; exit 1 ;;
            esac
        done
        if [[ "$benchmark" == "add-kernel" ]]; then
            target="cuda_elementwise_benchmark"
        else
            target="cuda_matmul_benchmark"
        fi
        cmake --build "$BUILD_DIR" -j --target "$target"
        if [[ "$benchmark_arg_count" -gt 0 ]]; then
            "$BUILD_DIR/$target" "${benchmark_args[@]}"
        else
            "$BUILD_DIR/$target"
        fi
        ;;
    *) usage; exit 1 ;;
esac
