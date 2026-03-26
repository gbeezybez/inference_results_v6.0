
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

GPU_ARCH=${GPU_ARCH:-"sm_100"}  # B200: sm_100

echo "============================================"
echo "Dynamic Bias Router v3 - Build"
echo "GPU_ARCH=${GPU_ARCH}"
echo "============================================"

build_cuda() {
    echo ""
    echo "Compiling CUDA kernel (v3)..."
    nvcc -O3 -std=c++17 \
         -gencode arch=compute_100,code=${GPU_ARCH} \
         -gencode arch=compute_100,code=compute_100 \
         --shared -Xcompiler -fPIC \
         -o libdynamic_bias_router_v3.so \
         dynamic_bias_router_v3.cu
    echo "  -> libdynamic_bias_router_v3.so"
}

build_triton() {
    echo ""
    echo "Triton kernels: no compilation needed (JIT at runtime)"
    echo "Verifying dependencies..."
    python3 -c "import triton; print(f'  Triton: {triton.__version__}')" 2>/dev/null || {
        echo "  ERROR: triton not found. Install: pip install triton"
        exit 1
    }
    python3 -c "import torch; print(f'  PyTorch: {torch.__version__}')"
    echo "  Ready."
}

run_tests() {
    echo ""
    echo "Running tests..."
    cd "$SCRIPT_DIR"
    python3 test_router.py --all
}

case "${1:-all}" in
    cuda)
        build_cuda
        ;;
    triton)
        build_triton
        ;;
    test)
        build_triton
        run_tests
        ;;
    all)
        build_triton
        build_cuda
        echo ""
        echo "Build complete. Run: ./build.sh test"
        ;;
    *)
        echo "Usage: $0 {all|cuda|triton|test}"
        exit 1
        ;;
esac
