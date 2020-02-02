#include "scatter.cuh"

#define TB_SIZE 1024  // 2^9 (has to be power of 2)

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

__global__ void calculate_mean_per_key_kernel(int n, int k, int *d_keys, float3 *d_values, float3 *d_means, int *d_counts) {
    const int block = blockIdx.x;  // responsible for a mean of the corresponding key
    const int thread = threadIdx.x;
    __shared__ float3 sh_partial_sum[TB_SIZE];
    __shared__ int sh_partial_count[TB_SIZE];

    // Iterating over whole array in chunks of block size (all threads are active)
    for (int i = 0; i * blockDim.x < n; ++i) {  // blockDim.x == TB_SIZE
        const int pos = i * blockDim.x + thread;
        if (pos >= n) {
            return;
        }
        const int key = d_keys[pos];
        sh_partial_sum[thread] = key == block ? d_values[pos] : make_float3(0.0f, 0.0f, 0.0f);
        sh_partial_count[thread] = key == block ? 1 : 0;
        __syncthreads();
        // Reduce using shared memory (TB_SIZE is a power of 2),
        // better thread-index assignment, no modulo operator, less idle threads
        for (int s = 1; s < blockDim.x; s <<= 1) {  // s *= 2
            int index = 2 * s * thread;
            if (index < blockDim.x) {
                sh_partial_sum[index] += sh_partial_sum[index + s];
                sh_partial_count[index] += sh_partial_count[index + s];
            }
            __syncthreads();
        }
        // Write to global memory (no conflicts, because only one block is responsible for writing to given key position)
        // No need for another kernel runs of final reductions.
        if (thread == 0) {
            if (i == 0) {
                d_means[block] = sh_partial_sum[0];
                d_counts[block] = sh_partial_count[0];
            }
            else {
                d_means[block] += sh_partial_sum[0];    
                d_counts[block] += sh_partial_count[0];
            }
        }
        __syncthreads();
    }
    // Finally, divide sum by count
    if (thread == 0) {
        d_means[block] /= d_counts[block];
    }
}

void calculate_mean_per_key_gather(int n, int k, int *d_keys, float3 *d_values, float3 *d_means) {
    int *d_counts;
    cudaMalloc(&d_counts, k*sizeof(int));
    cudaMemset(d_counts, 0, k*sizeof(int));
    calculate_mean_per_key_kernel<<<k, TB_SIZE>>> (n, k, d_keys, d_values, d_means, d_counts);
    cudaDeviceSynchronize();
}