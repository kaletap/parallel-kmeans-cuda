#include "scatter.cuh"

#define TB_SIZE 1024

using std::cout;
using std::endl;
using std::vector;

void print(float3 point) {
    cout << "float3(" << point.x << ", " << point.y << ", " << point.z << ") ";
}

void println(float3 point) {
    cout << "float3(" << point.x << ", " << point.y << ", " << point.z << ")" << endl;
}

void print(int a) {
    cout << a << " ";
}

__device__ __host__ float3 operator+(const float3 &a, const float3 &b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline __host__ __device__ void operator+=(float3 &a, float3 b)
{
    a.x += b.x; a.y += b.y; a.z += b.z;
}

inline __host__ __device__ float3 operator/(const float3 &a, const int b) {
    return make_float3(a.x / b, a.y / b, a.z / b);
}

inline __host__ __device__ void operator/=(float3 &a, const int b) {
    if (b != 0) {
        a.x /= b; a.y /= b; a.z /= b;
    }
    else {
        printf("Zero division!\n");
    }     
}

__device__ void atomicAdd(float3 *d_val, float3 val) {
    atomicAdd(&((*d_val).x), val.x);
    atomicAdd(&((*d_val).y), val.y);
    atomicAdd(&((*d_val).z), val.z);
}

// Jaka jest intuicja co jest dobrym rozwiązaniem a co nie?
__global__ void calculate_mean_per_key_kernel(int n, int k, int *d_keys, float3 *d_values, float3 *d_means, int *d_counts) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;  // tid <-> position in d_values
    if (tid >=n) {
        return;
    }
    // Update sums and counts
    const int key = d_keys[tid];
    atomicAdd(d_means + key, d_values[tid]);
    atomicAdd(d_counts + key, 1);
}

__global__ void normalize_means(int k, float3 *d_means, int* d_counts) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= k) {
        return;
    }
    d_means[tid] /= d_counts[tid];
}

void calculate_mean_per_key_scatter(int n, int k, int *d_keys, float3 *d_values, float3 *d_means) {
    int *d_counts;
    cudaMalloc(&d_counts, k*sizeof(int));
    cudaMemset(d_counts, 0, k*sizeof(int));
    const int n_blocks = (n + TB_SIZE - 1) / TB_SIZE;
    calculate_mean_per_key_kernel<<<n_blocks, TB_SIZE>>> (n, k, d_keys, d_values, d_means, d_counts);
    cudaDeviceSynchronize();
    const int n_blocks_normalization = (k + TB_SIZE - 1) / TB_SIZE;   // probably just one block (if k <= 1024)
    normalize_means<<<n_blocks_normalization, TB_SIZE>>> (k, d_means, d_counts);
    cudaDeviceSynchronize();
}