#ifndef SCATTER
#define SCATTER

#include <cstdlib>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cassert>
#include <numeric>
#include <stdio.h>


void calculate_mean_per_key_scatter(int n, int k, int *d_keys, float3 *d_values, float3 *d_means);

void print(float3 point);

void println(float3 point);

void print(int a);

#endif // SCATTER