//
// Created by premsai chinthamreddy on 26/10/18.
//

#include "vec3d.h"
#include <vector>
#include <algorithm>

vec3d::vec3d(int N) {
    arr.resize(N);
    fill(arr.begin(), arr.end(), 0.0);
}

void vec3d::set(int x, int y, int z, float val){
    int idx = N * N * z + x * N + j + 1;
    this->arr[idx] = val;
}

float vec3d::get(int x,int y, int z) {
    int idx = N * N * z + x * N + j + 1;
    return this->arr[idx];
}