#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="build"
CLEAN=0
METAL=0

usage() {
    echo "Usage: ./build.sh [--clean] [--metal]"
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

if [[ "$METAL" -eq 1 ]]; then
    cmake -S . -B "$BUILD_DIR" -DCITRIUS_ENABLE_METAL=ON
else
    cmake -S . -B "$BUILD_DIR" -DCITRIUS_ENABLE_METAL=OFF
fi

cmake --build "$BUILD_DIR"
