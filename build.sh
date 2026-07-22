#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="build"
CLEAN=0
METAL=0
CUDA=0
CUDA_CUBLAS=1
BUILD_TYPE="Release"

usage() {
    echo "Usage: ./build.sh [--clean] [--debug] [--metal] [--cuda|--cuda-reference]"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --clean)
            CLEAN=1
            shift
            ;;
        --debug)
            BUILD_TYPE="Debug"
            shift
            ;;
        --metal)
            METAL=1
            shift
            ;;
        --cuda)
            CUDA=1
            shift
            ;;
        --cuda-reference)
            CUDA=1
            CUDA_CUBLAS=0
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            usage
            exit 1
            ;;
    esac
done

if [[ "$CLEAN" -eq 1 ]]; then
    rm -rf "$BUILD_DIR"
fi

cmake -S . -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DCITRIUS_ENABLE_METAL=$([[ "$METAL" -eq 1 ]] && echo ON || echo OFF) \
    -DCITRIUS_ENABLE_CUDA=$([[ "$CUDA" -eq 1 ]] && echo ON || echo OFF) \
    -DCITRIUS_CUDA_USE_CUBLAS=$([[ "$CUDA_CUBLAS" -eq 1 ]] && echo ON || echo OFF)

cmake --build "$BUILD_DIR" -j
