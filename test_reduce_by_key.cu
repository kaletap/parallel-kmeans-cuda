#include "reduce_by_key.cuh"

using namespace std;

int main() {
    int N = 2000;
    int k = 3;
    size_t bytes_k = N * sizeof(int);
    size_t bytes_v = N * sizeof(float3);

    // Host data
    vector<float3> h_v(N);
    vector<int> h_k(N);
    vector<float3> h_out(k);

    for (int i = 0; i < N; ++i) {
        h_k[i] = i % 2;
        h_v[i] = make_float3(1.0, 1.0, 1.0);
    }

    for (int i = 0; i < N; ++i) {
        cout << h_k[i] << " ";
    }
    cout << endl;

    for (int i = 0; i < N; ++i) {
        print(h_v[i]);
    }
    cout << endl;

    // Allocate device memory and copy to device
    int *d_k;
    float3 *d_v, *d_v_r, *d_out;
    cudaMalloc(&d_k, bytes_k);
    cudaMalloc(&d_v, bytes_v);
    cudaMalloc(&d_v_r, k*bytes_v);
    cudaMalloc(&d_out, k*sizeof(float3));
    cudaMemcpy(d_k, h_k.data(), bytes_k, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, h_v.data(), bytes_v, cudaMemcpyHostToDevice);

    my_reduce_by_key(N, k, d_k, d_v, d_v_r, d_out);

    cudaMemcpy(h_out.data(), d_out, k*sizeof(float3), cudaMemcpyDeviceToHost);
    cout << "Sums by keys:" << endl;
    for (int i = 0; i < k; ++i) {
        print(h_out[i]);
    }
    cout << endl;
}