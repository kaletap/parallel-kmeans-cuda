#include <vector>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <curand.h>
#include <cassert>
#include <bits/stdc++.h> 

#include "gather.cuh"

#define OUT_FILE "labels.txt"
#define DEFAULT_K 3

//using namespace std;  you are swamped with errors after using namespace std
using namespace thrust;
using std::vector;
using std::cout;
using std::cin;
using std::endl;


template<typename T>
void println(device_vector<T> d_v) {
    host_vector<T> h_v = d_v;
    for (int i = 0; i < h_v.size(); ++i) {
        print(h_v[i]);
    }
    cout << endl;
}

__device__ __host__ float euklidian_distance_squared(const float3 a, const float3 &b) {
    return (a.x - b.x)*(a.x - b.x) + (a.y - b.y)*(a.y - b.y) + (a.z - b.z)*(a.z - b.z);
}

struct point_label : public thrust::unary_function<float3, int> {
    float3 *d_means;
    int k;
    __device__ __host__ point_label(float3 *_d_means, int _k) {
        d_means = _d_means;
        k = _k;
    }
    __device__ __host__  int operator()(float3 point) {
        float min_distance = FLT_MAX;
        int label = 0;
        for (int i = 0; i < k; ++i) {
            const float distance = euklidian_distance_squared(point, d_means[i]);
            if (distance < min_distance) {
                min_distance = distance;
                label = i;
            }
        }
        return label;
    }
};

struct squared_distance_functor {
    __device__ __host__ float operator()(tuple<float3, float3> a) {
        return euklidian_distance_squared(get<0>(a), get<1>(a));
    }
};

host_vector<int> kmeans(vector<float3> points, int k, int max_iter = 100, float eps = 1.0e-3) {
    const int n = points.size();
    
    device_vector<float3> d_points(points.begin(), points.end());
    device_vector<float3> means(points.begin(), points.begin() + k);
    device_vector<float3> old_means(k);

    device_vector<int> labels(n);  // label of each point

    // Creating raw pointers from thrust vectors
    int *d_labels_ptr = thrust::raw_pointer_cast(&labels[0]);
    float3 *d_points_ptr = thrust::raw_pointer_cast(&d_points[0]);
    float3* d_means_ptr = thrust::raw_pointer_cast(&means[0]);

    for (int i = 0; i < max_iter; ++i) {
        old_means = means;
        // Getting closest (in terms of euklidian distance) means
        transform(d_points.begin(), d_points.end(), labels.begin(), point_label(d_means_ptr, k));

        // Calculating mean of points per label
        calculate_mean_per_key_gather(n, k, d_labels_ptr, d_points_ptr, d_means_ptr);

        float mean_squared_distance = transform_reduce(
            make_zip_iterator(make_tuple(old_means.begin(), means.begin())),
            make_zip_iterator(make_tuple(old_means.end(), means.end())),
            squared_distance_functor(),
            0.0f,
            thrust::plus<float>()
        ) / k;

        if (mean_squared_distance < eps) {
            cout << "Converged after " << i + 1 << " iterations." << endl;
            cout << "Result save in a file " << OUT_FILE << "." << endl;
            break;
        }
    }
    
    return host_vector<int>(labels.begin(), labels.end());
}

/*
TODO: 
*/

int main(int argc, char **argv) {
    int k = argc > 1 ? atoi(argv[1]) : DEFAULT_K;
    vector<float3> points;
    int n;
    cin >> n;
    float x, y, z;
    for (int i = 0; i < n; ++i) {
        cin >> x >> y >> z;
        points.push_back(make_float3(x, y, z));
    }
    cout << "Running k-means with " << n << " points and k = " << k << endl;
    auto labels = kmeans(points, k);
    std::ofstream labels_file(OUT_FILE);

    for (int label : labels) {
        labels_file << label << " ";
    }

    return 0;
}
