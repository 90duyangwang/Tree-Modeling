//
// Created by premsai chinthamreddy on 22/10/18.
//

#include "voxel_reconstruction.h"
#include "vec3d.h"

#include <vector>
#include <cmath>


//vec3d create_voxel_grid(int img_width, int img_height){
//    // Given width and height of a image
//    // This function creates voxel grid which should be sparse
//    // Always use 25 X 25 X 25 grid for now (may be used to create specialized voxel grids for various image sizes)
//
//    int N = 25;
//    vec3d grid_alpha(N);
//    return grid_alpha;
//}




void initialize_voxel_densities(int img_h, int img_w, int N, vec3d &grid, vector<vector<int>> alpha) {
    // We project each cell onto the images
    // and calculate the average opacity by covering all the pixels met by the cell in a image
    // Initial value of the cell is the minimum of all the average opacity values in the respective images
    // the image opacities are calculated using the poisson matting algo
    // this is the initial alpha value of each cell
    // TODO: Initialization with diifferent images has to be handled, the final value shoudl be minimum of the average opacities
    vector<vector<int>> cells;
    for(int i=0; i<img_h; i++){
        for(int j=0; j<img_w; j++){
            cells = list_of_cells(img_h, img_w, N);
            for(auto& val: cells) {
                int prev =  grid.get(val[0], val[1], val[2]);
                grid.set(val[0], val[1], val[2], prev+alpha[i][j]);
            }
        }
    }

    int avg_pixels = (img_h/N) * (img_w/N);
    for(int i=0; i<N; i++) {
        for(int j=0; j<N; j++){
            for(int k=0; k<N; k++){
                float sum = grid.get(i,j,k);
                grid.set(i,j,k,sum/avg);
            }
        }
    }


}

vector<vector<int>> list_of_cells(pixel_h h, pixel_w w, int N){
    // Each pixel corresponds a voxel cell
    // If the image size is H X W
    // and the voxel is of size N X N X N
    // then H1 = H/N and W1 = W/N
    // Each pixel corresponds to (i/H1, j/W1, d<1..N>)
    // 1 <= i <= N
    // 1 <= j <= N
    // 1 <= d <= N

    // This function returns all the cells that are intersected by the ray through the pixel(i,j)

    float h1 = h/N;
    float w1 = w/N;
    vector<vector<int>> res;
    vector<int> temp;
    int yidx = ceil(w1);
    int xidx = ceil(h1);
    for(int k=0; k<N; k++) {
        temp.push_back(xidx);
        temp.push_back(yidx);
        temp.push_back(k);
        res.push_back(temp);
        temp.clear();
    }

    return res;
}

vec project_q_onto_plane(vec q, intercept qp, dim n) {
    // Equation of plane using Normal N(1,1,1,...,1) and intercept qp - n dimensions
    // using the equation of the plane in n-dimension find the projection of q on that plane
    // t = ad-ax + be -by + cf -cz /(a2 + b2 +c2)
    // N = (a,b,c)
    // qp = (qp,0,0) = (d,e,f)
    // q = (x,y,z)
    float sqsum = 0.0;
    vec N(n,1);
    for(int i=0; i<n; i++){
        sqsum += N[i]*N[i];
    }
    float neg_sum = 0.0;
    for(int i=0; i<n; i++){
        neg_sum += N[i]*q[i];
    }
    float t = (((qp - neg_sum)*1.0)/sqsum);
    vec res;
    for(int i=0; i<n; i++) {
        res.push_back(q[i] + t*N[i]);
    }

    return res;

}

bool convergence_condition() {

    // The convergence condition is whether the ti values that are not equal to 1 have changed
    // if the maximum change of all the ti values is less than a threshold return true
    // else return false
}

float calculate_q(float alpha){
    // This function simply calculates the qi value given a ti value of a cell
    // qi = log(ti)
    return log(1-alpha);
}


float calculate_t(float alpha) {
    // This funtion calculates the transparency of a cell given its alpha
    // ti = 1 - alphai
    return 1-alpha;
}


void alpha_calculation(int img_h, int img_w, vector<vector<float>> img_alpha) {

    //TODO: Update the algorithm for multiple images

    int N = 25;
    float delta;
    vec3d voxel(N);
    vec3d weights(N);
    vec3d del(N);
    initialize_voxel_densities(img_h, img_w, N, grid, img_alpha);
    vector<float> q;
    while(!convergence_condition()) {

        update_voxel_densities(di, w);
     //   for(auto &image: images){
            for(int imh=0; imh<img_h; imh++) {
                for(int imw=0; imw<img_w; imw++){
                    cells_traced = list_of_cells(imh, imw);
                    q.clear();
                    for(auto &cell : cells_traced){
                        q.push_back(calculate_q(voxel.get(cell[0],cell[1],cell[2])));
                    }
                    float qp = calculate_q(img_alpha[imh][imw]);
                    vec q1 = project_q_onto_plane(q, qp, N);
                    int idx = 0;
                    for(auto &cell : cells_traced) {
                        float temp = weights.get(cell[0],cell[1],cell[2])*( 1 - voxel.get(cell[0],cell[1],cell[2]) - exp(q1[idx++]));
                        voxel.set(cell[0],cell[1],cell[2], temp);
                        w += weights.get(cell[0],cell[1],cell[2]);
                        delta += temp;
                    }
                }
            }
       // }
        for(int i=0; i<N; i++){
            for(int j=0; j<N; j++){
                for(int k=0; k<N; k++){
                    float ti = voxel.get(i,j,k);
                    if(ti + delta/w > threshold){
                        voxel.set(i,j,k,1);
                    }else {
                        voxel.set(i, j, k, ti + delta / w);
                    }
                }
            }
        }

    }

}

