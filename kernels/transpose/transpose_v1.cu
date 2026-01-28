#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// 矩阵规模 1024 x 9600
#define M 1024
#define N 9600

#define TM 128
#define TN 64

template <int Tm, int Tn>
__global__ void transpose_v1(float *in, float *out)
{
    // 计算负责块
    int r0 = blockIdx.y * Tm;
    int c0 = blockIdx.x * Tn;
#pragma unroll
    for (int y = threadIdx.y; y < Tm; y += blockDim.y)
    {
        int r = r0 + y;
        if (r >= M)
            break;
#pragma unroll
        for (int x = threadIdx.x; x < Tn; x += blockDim.x)
        {
            int c = c0 + x;
            if (c < N)
                out[c * M + r] = in[r * N + c];
        }
    }
}

// ==========================================
// 1. CPU Naive 版本 (用于对比结果和时间)
// ==========================================
void cpu_transpose(float *in, float *out)
{
    for (int y = 0; y < M; y++)
    {
        for (int x = 0; x < N; x++)
        {
            // 输入: 行主序读取 in[y][x]
            // 输出: 转置写入 out[x][y]
            out[x * M + y] = in[y * N + x];
        }
    }
}

// 简单的初始化函数
void init_data(float *data)
{
    for (int i = 0; i < M * N; i++)
    {
        data[i] = (float)rand() / RAND_MAX;
    }
}

int main()
{
    size_t bytes = M * N * sizeof(float);
    printf("Matrix Size: %d x %d\n", M, N);

    // ------------------------------------------------
    // Part 1: Host (CPU) 内存分配与初始化
    // ------------------------------------------------
    float *h_in = (float *)malloc(bytes);         // 输入矩阵
    float *h_out = (float *)malloc(bytes);        // CPU计算结果
    float *h_gpu_result = (float *)malloc(bytes); // 存放GPU计算结果回传

    init_data(h_in);
    printf("Initializing data done.\n");

    // ------------------------------------------------
    // Part 2: 执行 CPU 转置 (基准测试)
    // ------------------------------------------------
    clock_t start_cpu = clock();

    cpu_transpose(h_in, h_out);

    clock_t end_cpu = clock();
    double cpu_time = (double)(end_cpu - start_cpu) / CLOCKS_PER_SEC;
    printf("CPU Time: %.4f s\n", cpu_time);

    // ------------------------------------------------
    // Part 3: GPU 实现部分
    // ------------------------------------------------
    // TODO 1: 分配设备端(GPU)内存 (cudaMalloc)
    float *d_in, *d_out;
    cudaMalloc((void **)&d_in, bytes);
    cudaMalloc((void **)&d_out, bytes);

    // TODO 2: 将数据从 Host 拷贝到 Device (cudaMemcpy HostToDevice)
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    // TODO 3: 配置 Kernel 参数 (dim3 block, dim3 grid)
    dim3 block(8, 32);
    dim3 grid((N + TN - 1) / TN, (M + TM - 1) / TM);

    // TODO 4: 调用 Kernel 函数

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    transpose_v1<TM, TN><<<grid, block>>>(d_in, d_out);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("GPU Kernel Time: %.4f ms (%.4f s)\n", milliseconds, milliseconds / 1000.0f);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    // TODO 5: 将结果从 Device 拷回 Host (cudaMemcpy DeviceToHost)
    cudaMemcpy(h_gpu_result, d_out, bytes, cudaMemcpyDeviceToHost);

    // TODO 6: 释放设备端内存 (cudaFree)
    cudaFree(d_in);
    cudaFree(d_out);

    // ------------------------------------------------
    // Part 4: 结果验证 & 输出
    // ------------------------------------------------
    int errors = 0;
    for (int i = 0; i < M * N; i++)
    {
        if (fabs(h_out[i] - h_gpu_result[i]) > 1e-5)
        {
            printf("Verification FAILED at index %d: CPU = %f, GPU = %f\n", i, h_out[i], h_gpu_result[i]);
            errors++;
            break;
        }
    }

    if (errors == 0)
    {
        // 1. 计算理论上的总读写字节数 (Naive 转置是 1次读 + 1次写)
        // Read: M * N * 4 bytes
        // Write: M * N * 4 bytes
        double total_bytes = 2.0 * M * N * sizeof(float);

        // 2. 计算有效带宽 (GB/s)
        // 带宽 = 总字节数 / 秒 / 10^9
        // milliseconds 是毫秒，要除以 1000 变秒
        double bandwidth_gbs = (total_bytes / (milliseconds / 1000.0)) / 1e9;

        // 3. 计算加速比
        // 注意单位统一：cpu_time 是秒，milliseconds 是毫秒
        double speedup = (cpu_time * 1000.0) / milliseconds;

        // ---------------- 打印格式化报告 ----------------
        printf("\ntranspose_v1 (%d*%d)\n", M, N);
        printf("M=%d, N=%d, block=(%d,%d), grid=(%d,%d)\n", M, N, block.x, block.y, grid.x, grid.y);
        printf("Correctness: PASS\n");
        printf("CPU time (single run): %.3f ms\n", cpu_time * 1000.0); // 转换为 ms
        printf("GPU kernel time (single run): %.3f ms\n", milliseconds);
        printf("Speedup (CPU / GPU kernel): %.2fx\n", speedup);
        printf("Estimated effective bandwidth (kernel): %.2f GB/s\n", bandwidth_gbs);
    }
    else
    {
        printf("Correctness: FAIL\n");
    }

    // 释放内存
    free(h_in);
    free(h_out);
    free(h_gpu_result);

    return 0;
}