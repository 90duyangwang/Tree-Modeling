//
// Created by premsai chinthamreddy on 26/10/18.
//

#ifndef IMAGEMATTING_VEC3D_H
#define IMAGEMATTING_VEC3D_H

#include <vector>

class vec3d {

public:
    vector<float> arr;
    vec3d(int N);
    void set(int x, int y, int z, float val);
    void get(int x, int y, int z);
};


#endif //IMAGEMATTING_VEC3D_H
