//
// Created by premsai chinthamreddy on 26/10/18.
//

#ifndef IMAGEMATTING_VEC3D_H
#define IMAGEMATTING_VEC3D_H

#include <vector>

class vec3d {

public:
    std::vector<float> arr;
    int N=0;
    vec3d(int N, float initval=0.0);
    void set(int x, int y, int z, float val);
    float get(int x, int y, int z);
    void update(float val);

};


#endif //IMAGEMATTING_VEC3D_H
