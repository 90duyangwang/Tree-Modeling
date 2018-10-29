//
// Created by premsai chinthamreddy on 26/10/18.
//

#include "vec3d.h"
#include <vector>
#include <algorithm>

vec3d::vec3d(int N, float initval) {
    this->N = N;
    arr.resize(N*N*N);
    fill(arr.begin(), arr.end(), initval);
}

void vec3d::set(int x, int y, int z, float val){
    int idx = N * N * z + x * N + y + 1;
    this->arr[idx] = val;
}

float vec3d::get(int x,int y, int z) {
    int idx = N * N * z + x * N + y + 1;
    return this->arr[idx];
}

void vec3d::update(float val) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                vec3d::set(i, j, k, val);
            }
        }
    }
}