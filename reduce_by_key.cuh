#ifndef REDUCE_BY_KEY
#define REDUCE_BY_KEY

#include <cstdlib>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cassert>
#include <numeric>

void my_reduce_by_key(int n, int k, int *d_keys, 
    float3* d_values, 
    float3 *d_almost_reduced_values, 
    float3 *d_output);

void print(float3 point);

void println(float3 point);

#endif // REDUCE_BY_KEY