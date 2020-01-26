#include "reduce_by_key.cuh"

#include <thrust/device_vector.h>

using namespace thrust;
using std::vector;
using std::cout;
using std::endl;

template<typename T>
void println(device_vector<T> d_v) {
    host_vector<T> h_v = d_v;
    for (int i = 0; i < h_v.size(); ++i) {
        print(h_v[i]);
    }
    cout << endl;
}

int main() {
    int N = 6;
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
    h_k[0] = 2;
    h_v[0] = make_float3(2.0, 1.5, 1.5);

    for (int i = 0; i < N; ++i) {
        cout << h_k[i] << " ";
    }
    cout << endl;

    for (int i = 0; i < N; ++i) {
        print(h_v[i]);
    }
    cout << endl;

    // Allocate device memory and copy to device
    int *d_k, *d_k_r;
    float3 *d_v, *d_v_r, *d_out;
    cudaMalloc(&d_k, bytes_k);
    cudaMalloc(&d_k_r, k*bytes_k);
    cudaMalloc(&d_v, bytes_v);
    cudaMalloc(&d_v_r, k*bytes_v);
    cudaMalloc(&d_out, k*sizeof(float3));
    cudaMemcpy(d_k, h_k.data(), bytes_k, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, h_v.data(), bytes_v, cudaMemcpyHostToDevice);

    // Test of sum reduce *****************************************************
    my_reduce_by_key(N, k, d_k, d_v, d_v_r, d_out);
    cudaMemcpy(h_out.data(), d_out, k*sizeof(float3), cudaMemcpyDeviceToHost);
    cout << "Sums by keys:" << endl;
    for (int i = 0; i < k; ++i) {
        print(h_out[i]);
    }
    cout << endl;

    // Test of count reduce and calculating means *****************************
    my_reduce_by_key(N, k, d_k, d_k_r, d_out);
    cudaMemcpy(h_out.data(), d_out, k*sizeof(float3), cudaMemcpyDeviceToHost);
    cout << "Means by keys:" << endl;
    for (int i = 0; i < k; ++i) {
        print(h_out[i]);
    }
    cout << endl;

    // Another test with thrust device pointers *******************************
    cout << endl;
    vector<float3> points({
        {1.2, 1.3, 1.4},
        {2.3, 2.4, 2.5},
        {2.3, 2.4, 2.4},
        {4.3, 2.4, 2.5},
        {2.3, 2.4, 2.5},
        {2.3, 2.4, 2.5},
    });
    int n = 6;
    for(int i = 0; i < n; ++i) {
        print(points[i]);
    }
    cout << endl;
    device_vector<float3> d_points(points.begin(), points.end());
    vector<int> h_labels({0, 1, 2, 1, 1, 1});
    device_vector<int> labels(h_labels.begin(), h_labels.end());
    vector<float3> means(k);

    // float3* d_points_ptr;
    // cudaMalloc(&d_points_ptr, n*sizeof(float3));
    // cudaMemcpy(d_points_ptr, points.data(), n*sizeof(float3), cudaMemcpyHostToDevice);
    // float3* d_means_ptr;
    // cudaMalloc(&d_means_ptr, n*sizeof(float3));
    // int* d_labels_ptr;
    // cudaMalloc(&d_labels_ptr, n*sizeof(int));
    // cudaMemcpy(d_labels_ptr, h_labels.data(), n*sizeof(int), cudaMemcpyHostToDevice);

    float3 *d_almost_reduced_values;
    cudaMalloc(&d_almost_reduced_values, n*k*sizeof(float3));
    int *d_almost_reduced_count;
    cudaMalloc(&d_almost_reduced_count, n*k*sizeof(int));

    // Creating raw pointers from thrust vectors
    int *d_labels_ptr = thrust::raw_pointer_cast(&labels[0]);
    float3 *d_points_ptr = thrust::raw_pointer_cast(&d_points[0]);
    float3* d_means_ptr = thrust::raw_pointer_cast(&means[0]);

    my_reduce_by_key(n, k, d_labels_ptr, d_points_ptr, d_almost_reduced_values, d_means_ptr);  // careful with arguments order
    cout << "Sum of points by mean: ";
    cudaMemcpy(means.data(), d_means_ptr, k*sizeof(float3), cudaMemcpyDeviceToHost);
    for (int i = 0; i < k; ++i) {
        print(means[i]);
    }
    cout << endl;
    // Dividing by keys on labels (hacky implementation fo my_reduce_by_key for table of ints)
    // hence passing d_means_ptr as output
    my_reduce_by_key(n, k, d_labels_ptr, d_almost_reduced_count, d_means_ptr);
    cudaMemcpy(means.data(), d_means_ptr, k*sizeof(float3), cudaMemcpyDeviceToHost);
    for (int i = 0; i < k; ++i) {
        print(means[i]);
    }
    cout << endl;

}