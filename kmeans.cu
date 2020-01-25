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

//using namespace std;  you are swamped with errors after using namespace std
using namespace thrust;
using std::vector;
using std::cout;
using std::endl;

float3 operator+(const float3 &a, const float3 &b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

void println(float3 point) {
    cout << "float3(" << point.x << ", " << point.y << ", " << point.z << ")" << endl;
}

float euklidian_distance_squared(const float3 a, const float3 &b) {
    return (a.x - b.x)*(a.x - b.x) + (a.y - b.y)*(a.y - b.y) + (a.z - b.z)*(a.z - b.z);
}

struct closer_point_tuple {
    float3 origin;
    closer_point_tuple(float3 _origin) {
        origin = _origin;
    }
    tuple<int, float3> operator()(tuple<int, float3> a, tuple<int, float3> b) {
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
    host_vector<float3> *means;
    point_label(host_vector<float3> *_means) {
        means = _means;
    }
    int operator()(float3 point) {
        counting_iterator<int> begin(0);
        counting_iterator<int> end(means->size());
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
template<class T>
struct value_at : public thrust::unary_function<int, T> {
    host_vector<T> *vec;
    value_at(host_vector<T> *_vec) {
        vec = _vec;
    }
    T operator()(int i) {
        return (*vec)[i];
    }
};

struct compare_by_label {
    bool operator()(const tuple<int, float3> &a, const tuple<int, float3> b) {
        return get<0>(a) < get<0>(b);
    }
};

struct squared_distance_functor {
    bool operator()(tuple<float3, float3> a) {
        return euklidian_distance_squared(get<0>(a), get<1>(a));
    }
};


vector<int> kmeans(vector<float3> points, int k, int max_iter = 100, float eps = 1.0e-3) {
    const int n = points.size();
    
    host_vector<float3> d_points(points.begin(), points.end());
    host_vector<float3> means(points.begin(), points.begin() + k);
    host_vector<float3> old_means(k);

    host_vector<int> _mean_keys(k);  // useless variable
    host_vector<int> labels(n);  // label of each point

    for (int i = 0; i < max_iter; ++i) {
        old_means = means;
        // Getting closest (in terms of euklidian distance) means
        transform(d_points.begin(), d_points.end(), labels.begin(), point_label(&means));

        // Calculating average on labels
        //   1. Sort by keys (slow - TODO: find a way to improve)
        sort(
            make_zip_iterator(make_tuple(labels.begin(), d_points.begin())), 
            make_zip_iterator(make_tuple(labels.end(), d_points.end())),
            compare_by_label()
        );

        //   2. Reduce by keys
        auto point_values_begin = make_transform_iterator(labels.begin(), value_at<float3>(&d_points));

        //  TODO: improve, there are exactly k labels BUT randomly shuffled (so reduce_by_key does not work without sorting first): 
        // https://thrust.github.io/doc/group__reductions_ga1fd25c0e5e4cc0a6ab0dcb1f7f13a2ad.html#ga1fd25c0e5e4cc0a6ab0dcb1f7f13a2ad
        reduce_by_key(labels.begin(), labels.end(), point_values_begin, _mean_keys.begin(), means.begin());

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