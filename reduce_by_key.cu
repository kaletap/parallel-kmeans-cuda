#include "reduce_by_key.cuh"

#define TB_SIZE 256
#define MAX_K 3

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

/****************************FLOAT3 VERSION*********************************/

__global__ void my_reduce_by_key_kernel(int n, int k, int *keys, float3 *values, float3 *almost_reduced_values) {
    __shared__ float3 partial_sum[TB_SIZE][MAX_K];

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) {
        return;
    }
    // Initalize shared memory to zero!
    for (int i = 0; i < k; ++i) {
        partial_sum[threadIdx.x][i] = make_float3(0, 0, 0);
    }
    const int key = keys[tid];  // value from 0 to k-1
    // Load elements into shared memory
    partial_sum[threadIdx.x][key] = values[tid];
    __syncthreads();

    for (int s = 1; s < blockDim.x; s <<= 1) {
        if (threadIdx.x % (2*s) == 0) {
            for (int i = 0; i < k; ++i) {
                partial_sum[threadIdx.x][i] += partial_sum[threadIdx.x + s][i];
            }
        }
        __syncthreads();
    }

    // Frist thread in a block writes to main memory
    if (threadIdx.x == 0) {
        for (int i = 0; i < k; ++i) {
            const int pos = blockIdx.x * k + i;
            almost_reduced_values[pos] = partial_sum[0][i];
        }
    }
}

// run at the end of the reduce by key with only one block
__global__ void sum_reduce(int n, int k, float3 *d_almost_reduces_values, float3 *output) {
    __shared__ float3 partial_sum[TB_SIZE][MAX_K];
    const int tid = threadIdx.x;
    for (int i = 0; i < k; ++i) {
        const int pos = tid * k + i;
        partial_sum[tid][i] = d_almost_reduces_values[pos];
    }
    __syncthreads();
    for (int s = 1; s < blockDim.x; s <<= 1) {
        if (tid % (2*s) == 0) {
            for (int i = 0; i < k; ++i) {
                partial_sum[tid][i] += partial_sum[threadIdx.x + s][i];
            }
        }
        __syncthreads();
    }
    if (tid == 0) {
        for (int i = 0; i < k; ++i) {
            output[i] = partial_sum[0][i];
        }
    }
}

void my_reduce_by_key(int n, int k, int *d_keys, 
    float3* d_values, 
    float3 *d_almost_reduced_values, 
    float3 *d_output) {
const int N_BLOCKS = (n + TB_SIZE - 1) / TB_SIZE;
my_reduce_by_key_kernel<<<N_BLOCKS, TB_SIZE>>> (n, k, d_keys, d_values, d_almost_reduced_values);
// if (n > TB_SIZE)
cudaDeviceSynchronize();
sum_reduce<<<1, TB_SIZE>>> (N_BLOCKS, k, d_almost_reduced_values, d_output);
cudaDeviceSynchronize();
}

/****************************INT VERSION*********************************/
__global__ void my_reduce_by_key_kernel(int n, int k, int *keys, int *almost_reduced_values) {
    __shared__ int partial_sum[TB_SIZE][MAX_K];

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) {
        return;
    }
    // Initalize shared memory to zero!
    for (int i = 0; i < k; ++i) {
        partial_sum[threadIdx.x][i] = 0;
    }
    const int key = keys[tid];  // value from 0 to k-1
    // Load elements into shared memory
    partial_sum[threadIdx.x][key] = 1;  // 1 because we want to count here
    __syncthreads();

    for (int s = 1; s < blockDim.x; s <<= 1) {
        if (threadIdx.x % (2*s) == 0) {
            for (int i = 0; i < k; ++i) {
                partial_sum[threadIdx.x][i] += partial_sum[threadIdx.x + s][i];
            }
        }
        __syncthreads();
    }

    // Frist thread in a block writes to main memory
    if (threadIdx.x == 0) {
        for (int i = 0; i < k; ++i) {
            const int pos = blockIdx.x * k + i;
            almost_reduced_values[pos] = partial_sum[0][i];
        }
    }
}

// run at the end of the reduce by key with only one block
__global__ void sum_reduce(int n, int k, int *d_almost_reduces_values, float3 *output) {
    __shared__ int partial_sum[TB_SIZE][MAX_K];
    const int tid = threadIdx.x;  // there are exactly N_BLOCKS threads and only one block at this point
    // Initalizing shared memory to zero!
    for (int i = 0; i < k; ++i) {
        partial_sum[tid][i] = 0;
    }
    __syncthreads();
    if (tid < n) {
        for (int i = 0; i < k; ++i) {
            const int pos = tid * k + i;
            partial_sum[tid][i] = d_almost_reduces_values[pos];
        }
    }
    __syncthreads();
    for (int s = 1; s < blockDim.x; s <<= 1) {
        if (tid % (2*s) == 0 && tid + s < n) {
            for (int i = 0; i < k; ++i) {
                partial_sum[tid][i] += partial_sum[tid + s][i];
            }
        }
        __syncthreads();
    }
    if (tid == 0) {
        for (int i = 0; i < k; ++i) {
            output[i] /= partial_sum[0][i];
            // output[i] = make_float3(partial_sum[0][i], partial_sum[0][i], partial_sum[0][i]);
        }
    }
}

void my_reduce_by_key(int n, int k, int *d_keys, 
                      int *d_almost_reduced_values, 
                      float3 *d_output) {
    const int N_BLOCKS = (n + TB_SIZE - 1) / TB_SIZE;
    my_reduce_by_key_kernel<<<N_BLOCKS, TB_SIZE>>> (n, k, d_keys, d_almost_reduced_values);
    // if (n > TB_SIZE)
    cudaDeviceSynchronize();
    sum_reduce<<<1, TB_SIZE>>> (N_BLOCKS, k, d_almost_reduced_values, d_output);
    cudaDeviceSynchronize();
}
