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

#include "reduce_by_key.cuh"

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

struct closer_point_tuple {
    float3 origin;
    __device__ __host__ closer_point_tuple(float3 _origin) {
        origin = _origin;
    }
    __device__ __host__ tuple<int, float3> operator()(tuple<int, float3> a, tuple<int, float3> b) {
        const float3 point_a = get<1>(a);
        const float3 point_b = get<1>(b);
        if (euklidian_distance_squared(origin, point_a) < euklidian_distance_squared(origin, point_b)) {
            return a;
        }
        else {
            return b;
        }
    }
};

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
        // TODO: improve - write custom kernel
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

// TODO: find better name
// template<class T>
// struct value_at : public thrust::unary_function<int, T> {
//     device_vector<T> *vec;
//     value_at(device_vector<T> *_vec) {
//         vec = _vec;
//     }
//     T operator()(int i) {
//         return (*vec)[i];
//     }
// };

struct compare_by_label {
    __device__ __host__ bool operator()(const tuple<int, float3> &a, const tuple<int, float3> b) {
        return get<0>(a) < get<0>(b);
    }
};

struct squared_distance_functor {
    __device__ __host__ bool operator()(tuple<float3, float3> a) {
        return euklidian_distance_squared(get<0>(a), get<1>(a));
    }
};

host_vector<int> kmeans(vector<float3> points, int k, int max_iter = 100, float eps = 1.0e-3) {
    const int n = points.size();
    
    device_vector<float3> d_points(points.begin(), points.end());
    device_vector<float3> means(points.begin(), points.begin() + k);  // TODO: it is necessary for means to be device vectors?
    device_vector<float3> old_means(k);

    device_vector<int> labels(n);  // label of each point

    float3 *d_almost_reduced_values;
    cudaMalloc(&d_almost_reduced_values, n*k*sizeof(float3));
    int *d_almost_reduced_count;
    cudaMalloc(&d_almost_reduced_count, n*k*sizeof(int));

    // Creating raw pointers from thrust vectors
    int *d_labels_ptr = thrust::raw_pointer_cast(&labels[0]);
    float3 *d_points_ptr = thrust::raw_pointer_cast(&d_points[0]);
    float3* d_means_ptr = thrust::raw_pointer_cast(&means[0]);

    // cout << "Points: ";
    // println(d_points);
    for (int i = 0; i < max_iter; ++i) {
        cout << "***Iteration number " << i << "***" << endl;
        old_means = means;
        // Getting closest (in terms of euklidian distance) means
        // TODO: write custom kernel for this
        transform(d_points.begin(), d_points.end(), labels.begin(), point_label(d_means_ptr, k));

        // Calculating sum on labels
        my_reduce_by_key(n, k, d_labels_ptr, d_points_ptr, d_almost_reduced_values, d_means_ptr);

        // Dividing by keys on labels (hacky implementation of my_reduce_by_key for table of ints)
        // hence passing d_means_ptr as output
        my_reduce_by_key(n, k, d_labels_ptr, d_almost_reduced_count, d_means_ptr);

        // cout << "Means: ";
        // println(means);
        // cout << "Labels: ";
        // println(labels);

        // TODO: is it worth it?
        float squared_distance = transform_reduce(
            make_zip_iterator(make_tuple(old_means.begin(), means.begin())),
            make_zip_iterator(make_tuple(old_means.end(), means.end())),
            squared_distance_functor(),
            0.0f,
            thrust::plus<float>()
        );

        if (squared_distance < eps) {
            cout << "***End of iterations***" << endl;
            cout << "Convered after " << i + 1 << " iterations." << endl;
            break;
        }
    }
    
    return host_vector<int>(labels.begin(), labels.end());
}

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