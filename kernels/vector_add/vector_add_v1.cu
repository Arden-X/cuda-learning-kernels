// kernels/vector_add/vector_add_v1.cu

#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>

// =====================
// 2. Macros & constants
// =====================
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif

#define CUDA_CHECK(call)                                               \
    do                                                                 \
    {                                                                  \
        cudaError_t err = (call);                                      \
        if (err != cudaSuccess)                                        \
        {                                                              \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n",             \
                         __FILE__, __LINE__, cudaGetErrorString(err)); \
            std::exit(1);                                              \
        }                                                              \
    } while (0)

// =====================
// 3. CUDA kernel(s)
// =====================
__global__ void vector_add_v1_kernel(int *A,
                                     int *B,
                                     int *C,
                                     int N)
{

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int vec_threads = N >> 2;
    int tail_items = N & 3;
    if (idx >= vec_threads + tail_items)
        return;
    if (idx < N / 4)
    {
        int4 a = reinterpret_cast<int4 *>(A)[idx];
        int4 b = reinterpret_cast<int4 *>(B)[idx];

        int4 c;
        c.x = a.x + b.x;
        c.y = a.y + b.y;
        c.z = a.z + b.z;
        c.w = a.w + b.w;
        reinterpret_cast<int4 *>(C)[idx] = c;
    }
    else
    {
        int start = (N / 4) * 4;
        // 计算当前线程应该负责哪个尾部元素
        int tail_idx = start + (idx - N / 4);
        if (tail_idx < N)
        {
            C[tail_idx] = A[tail_idx] + B[tail_idx];
        }
    }
}

// =====================
// 4. CPU reference
// =====================
void vector_add_cpu_ref(const int *A,
                        const int *B,
                        int *C,
                        int N)
{
    for (int i = 0; i < N; ++i)
    {
        C[i] = A[i] + B[i];
    }
}

// =====================
// 5. Utility
// =====================
void init_data(int *A, int *B, int N)
{
    for (int i = 0; i < N; ++i)
    {
        A[i] = i;
        B[i] = i * 2;
    }
}

bool check_equal(const int *ref, const int *out, int N)
{
    for (int i = 0; i < N; ++i)
    {
        if (ref[i] != out[i])
        {
            std::printf("Mismatch at %d: ref=%d out=%d\n", i, ref[i], out[i]);
            return false;
        }
    }
    return true;
}

// =====================
// 6. main
// =====================
int main(int argc, char **argv)
{
    // ---- 6.1 parameters ----
    int N = (argc >= 2) ? std::atoi(argv[1]) : 16 * 1024 * 1024;
    size_t bytes = static_cast<size_t>(N) * sizeof(int);

    // ---- 6.2 host alloc + init ----
    int *h_A = (int *)std::malloc(bytes);
    int *h_B = (int *)std::malloc(bytes);
    int *h_C_gpu = (int *)std::malloc(bytes);
    int *h_C_cpu = (int *)std::malloc(bytes);

    if (!h_A || !h_B || !h_C_gpu || !h_C_cpu)
    {
        std::fprintf(stderr, "Host malloc failed\n");
        return 1;
    }

    init_data(h_A, h_B, N);

    // ---- 6.3 device alloc + H2D ----
    int *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // ---- 6.4 launch config ----
    dim3 block(BLOCK_SIZE);
    dim3 grid((N + block.x * 4 - 1) / (block.x * 4));

    // ---- 6.5 warm-up ----
    vector_add_v1_kernel<<<grid, block>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // ---- 6.6 GPU kernel timing ----
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    const int iters = 50;
    CUDA_CHECK(cudaEventRecord(start));
    for (int it = 0; it < iters; ++it)
    {
        vector_add_v1_kernel<<<grid, block>>>(d_A, d_B, d_C, N);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float gpu_ms_total = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_ms_total, start, stop));
    float gpu_ms = gpu_ms_total / iters;

    // ---- 6.7 D2H ----
    CUDA_CHECK(cudaMemcpy(h_C_gpu, d_C, bytes, cudaMemcpyDeviceToHost));

    // ---- 6.8 CPU timing ----
    auto cpu_start = std::chrono::high_resolution_clock::now();
    vector_add_cpu_ref(h_A, h_B, h_C_cpu, N);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

    // ---- 6.9 correctness ----
    bool ok = check_equal(h_C_cpu, h_C_gpu, N);

    // ---- 6.10 report ----
    // For vector add: read A + read B + write C = 3 * sizeof(int) bytes per element
    double bytes_moved = 3.0 * static_cast<double>(bytes);
    double gbps = (bytes_moved / 1e9) / (gpu_ms / 1e3);

    std::printf("vector_add_v1\n");
    std::printf("N=%d, block=%d, grid=%d\n", N, (int)block.x, (int)grid.x);
    std::printf("Correctness: %s\n", ok ? "PASS" : "FAIL");
    std::printf("CPU time (single run): %.3f ms\n", cpu_ms);
    std::printf("GPU kernel time (avg over %d iters): %.3f ms\n", iters, gpu_ms);
    std::printf("Speedup (CPU / GPU kernel): %.2fx\n", cpu_ms / gpu_ms);
    std::printf("Estimated effective bandwidth (kernel): %.2f GB/s\n", gbps);

    // ---- 6.11 cleanup ----
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    std::free(h_A);
    std::free(h_B);
    std::free(h_C_gpu);
    std::free(h_C_cpu);

    return ok ? 0 : 2;
}
