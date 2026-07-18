#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="build"
CLEAN=0
METAL=0
CUDA=0

usage() {
    echo "Usage: ./build.sh [--clean] [--metal] [--cuda]"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --clean)
            CLEAN=1
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
    -DCITRIUS_ENABLE_METAL=$([[ "$METAL" -eq 1 ]] && echo ON || echo OFF) \
    -DCITRIUS_ENABLE_CUDA=$([[ "$CUDA" -eq 1 ]] && echo ON || echo OFF)

cmake --build "$BUILD_DIR"
