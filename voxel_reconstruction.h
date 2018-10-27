//
// Created by premsai chinthamreddy on 22/10/18.
//

#ifndef IMAGEMATTING_VOXEL_RECONSTRUCTION_H
#define IMAGEMATTING_VOXEL_RECONSTRUCTION_H

typedef float delta;
typedef float weight;
typedef float coeff;
typedef float transparency;
typedef vector<float> vec;
typedef float intercept;
typedef int dim;


vector<vector<vector<int>>> create_voxel_grid();

void initialize_voxel_densities(delta di, weight w);

void list_of_cells();

void project_q_onto_plane();

void convergence_condition();

void alpha_calculation();

#endif //IMAGEMATTING_VOXEL_RECONSTRUCTION_H
