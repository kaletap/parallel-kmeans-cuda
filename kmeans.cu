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

//using namespace std;  you are swamped with errors after using namespace std
using namespace thrust;
using std::vector;
using std::cout;
using std::endl;

__device__ __host__ float3 operator+(const float3 &a, const float3 &b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

void println(float3 point) {
    cout << "float3(" << point.x << ", " << point.y << ", " << point.z << ")" << endl;
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
    device_vector<float3> *means;
    __device__ __host__ point_label(device_vector<float3> *_means) {
        means = _means;
    }
    __device__ __host__  int operator()(float3 point) {
        counting_iterator<int> begin(0);
        counting_iterator<int> end((*means).size());
        tuple<int, float3> init = make_tuple(0, (*means)[0]);
        tuple<int, float3> closest_mean_tuple;
        closest_mean_tuple = reduce(make_zip_iterator(make_tuple(begin, means->begin())),
                                    make_zip_iterator(make_tuple(end, means->end())),
                                    init,
                                    closer_point_tuple(point));
        return get<0>(closest_mean_tuple);
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

vector<int> kmeans(vector<float3> points, int k, int max_iter = 100, float eps = 1.0e-3) {
    const int n = points.size();
    
    device_vector<float3> d_points(points.begin(), points.end());
    device_vector<float3> means(points.begin(), points.begin() + k);  // TODO: it is necessary for means to be device vectors?
    device_vector<float3> old_means(k);

    device_vector<int> _mean_keys(k);  // useless variable
    device_vector<int> labels(n);  // label of each point

    float3 *d_almost_reduced_values;
    cudaMalloc(&d_almost_reduced_values, n*k*sizeof(float3));

    // Creating raw pointers from thrust vectors
    int *d_labels_ptr = thrust::raw_pointer_cast(&labels[0]);
    float3 *d_points_ptr = thrust::raw_pointer_cast(&d_points[0]);
    float3* d_means_ptr = thrust::raw_pointer_cast(&means[0]);

    for (int i = 0; i < max_iter; ++i) {
        old_means = means;
        // Getting closest (in terms of euklidian distance) means
        transform(d_points.begin(), d_points.end(), labels.begin(), point_label(&means));

        // Calculating average on labels
        my_reduce_by_key(n, k, d_labels_ptr, d_almost_reduced_values, d_points_ptr, d_means_ptr);

        float squared_distance = transform_reduce(
            make_zip_iterator(make_tuple(old_means.begin(), means.begin())),
            make_zip_iterator(make_tuple(old_means.end(), means.end())),
            squared_distance_functor(),
            0.0f,
            thrust::plus<float>()
        );

        if (squared_distance < eps) {
            break;
        }
    }
    
    return vector<int>(labels.begin(), labels.end());
}


int main() {
    vector<float3> points({
        {1.2, 1.3, 1.4},
        {2.3, 2.4, 2.5},
        {2.3, 2.4, 2.4},
        {4.3, 2.4, 2.5},
        {2.3, 2.4, 2.5},
        {2.3, 2.4, 2.5},
    });
    auto labels = kmeans(points, 3);
    for (int label : labels) {
        cout << label << endl;
    }
    cout << endl;
    return 0;
}