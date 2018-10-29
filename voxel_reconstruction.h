//
// Created by premsai chinthamreddy on 22/10/18.
//

#ifndef IMAGEMATTING_VOXEL_RECONSTRUCTION_H
#define IMAGEMATTING_VOXEL_RECONSTRUCTION_H

#include "vec3d.h"
#include <vector>

using namespace std;

vector<int> list_of_cells(int h, int w, int N);

void initialize_voxel_densities(int img_h, int img_w, int N, vec3d &grid, vector<vector<float>> &alpha);

vector<float> project_q_onto_plane(vector<float> &q, float qp, int n);

bool convergence_condition(vec3d& del, vec3d& grid, float threshold, int N);

float calculate_q(float alpha);

float calculate_t(float alpha);

vec3d alpha_calculation(int img_h, int img_w, vector<vector<float>> &img_alpha);

#endif //IMAGEMATTING_VOXEL_RECONSTRUCTION_H
