//
// Created by premsai chinthamreddy on 22/10/18.
//

#include "vec3d.h"
#include "voxel_reconstruction.h"
#include <vector>
#include <cmath>
#include <iostream>

using namespace std;

//vec3d create_voxel_grid(int img_width, int img_height){
//    // Given width and height of a image
//    // This function creates voxel grid which should be sparse
//    // Always use 25 X 25 X 25 grid for now (may be used to create specialized voxel grids for various image sizes)
//
//    int N = 25;
//    vec3d grid_alpha(N);
//    return grid_alpha;
//}



vector<int> list_of_cells(int h, int w, int N){
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
    vector<int> temp;
    int yidx = floor(w1);
    int xidx = floor(h1);
    temp.push_back(xidx);
    temp.push_back(yidx);

    return temp;
}


void initialize_voxel_densities(int img_h, int img_w, int N, vec3d &grid, vector<vector<float>> &alpha) {
    // We project each cell onto the images
    // and calculate the average opacity by covering all the pixels met by the cell in a image
    // Initial value of the cell is the minimum of all the average opacity values in the respective images
    // the image opacities are calculated using the poisson matting algo
    // this is the initial alpha value of each cell
    // TODO: Initialization with diifferent images has to be handled, the final value shoudl be minimum of the average opacities

    vector<int> cells;
//    cout << "# Rows: " << img_h << " # Cols: " << img_w << endl;
//    cout << "Initializing Voxels" << endl;
    for(int i=0; i<img_h; i++){
//        cout << " Processing Row: # " << i <<endl;
        for(int j=0; j<img_w; j++){
//            cout << "Processing Col: # "<< j << endl;
            cells = list_of_cells(i, j, N);
            for(int k=0; k<N; k++) {
                float prev =  grid.get(cells[0], cells[1], k);
                grid.set(cells[0], cells[1], k, prev+alpha[i][j]);
            }
        }
    }
//    cout << "After First Loop" << endl;
    int avg_pixels = (img_h/N) * (img_w/N);
    for(int i=0; i<N; i++) {
        for(int j=0; j<N; j++){
            for(int k=0; k<N; k++){
                float sum = grid.get(i,j,k);
                grid.set(i,j,k,sum/avg_pixels);
            }
        }
    }

    cout << "completed voxel initialization" << endl;
}

vector<float> project_q_onto_plane(vector<float> &q, float qp, int n) {
    // Equation of plane using Normal N(1,1,1,...,1) and intercept qp - n dimensions
    // using the equation of the plane in n-dimension find the projection of q on that plane
    // t = ad-ax + be -by + cf -cz /(a2 + b2 +c2)
    // N = (a,b,c)
    // qp = (qp,0,0) = (d,e,f)
    // q = (x,y,z)
    float sqsum = 0.0;
    vector<float> N(n,1);
    for(int i=0; i<n; i++){
        sqsum += N[i]*N[i];
    }
    float neg_sum = 0.0;
    for(int i=0; i<n; i++){
        neg_sum += N[i]*q[i];
    }
    float t = (((qp - neg_sum)*1.0)/sqsum);
    vector<float> res;
    for(int i=0; i<n; i++) {
        res.push_back(q[i] + t*N[i]);
    }

    return res;

}

bool convergence_condition(vec3d& del, vec3d& grid, float threshold, int N) {

    // The convergence condition is whether the ti values that are not equal to 1 have changed
    // if the maximum change of all the ti values is less than a threshold return true
    // else return false
    float max_change = 0.0;
    for(int i=0; i<N; i++){
        for(int j=0; j<N; j++){
            for(int k=0; k<N; k++){
                if(grid.get(i,j,k) != 1.0){
                    if(max_change < del.get(i,j,k)){
                        max_change = del.get(i,j,k);
                    }
                }
            }
        }
    }
    return (max_change < threshold);
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

void vec_print(vector<float> v){
    cout << "Printing Vector: ";
    for(auto &n : v){
        cout << n <<" , ";
    }
    cout << endl;
}

vec3d alpha_calculation(int img_h, int img_w, vector<vector<float>> &img_alpha) {

    //TODO: Update the algorithm for multiple images
    //TODO: Update the Weight matrix for now we are considering equal weights to all cells wi = 1
    //TODO: determine the threshold to make the transparencies to 1
    cout << "Alpha image: Rows: # " << img_alpha.size() << "  Cols: # " << img_alpha[0].size() << endl;
//    for(auto &inner: img_alpha){
//        vec_print(inner);
//    }

    int N = 25;
    vec3d voxel(N); // voxels save the transparency value (ti s not alpha )
    vec3d weights(N,1);
    vec3d del(N,0.0);
    initialize_voxel_densities(img_h, img_w, N, voxel, img_alpha);
    vector<float> q;
    int conv = 0;
    while(conv++ < 2) { //!convergence_condition(del, voxel, 0.4, N)) {
        del.update(0.0);
        float w = 0.0;
     //   for(auto &image: images){
            for(int imh=0; imh<img_h; imh++) {
                for(int imw=0; imw<img_w; imw++){
                    vector<int> cells = list_of_cells(imh, imw, N);
                    for(int k=0; k<N; k++){
                        q.push_back(calculate_q(voxel.get(cells[0],cells[1],k)));
                    }
                    float qp = calculate_q(img_alpha[imh][imw]);
                    vector<float> q1 = project_q_onto_plane(q, qp, N);
                    q.clear();
                    int idx = 0;
                    for(int k=0 ;k<N; k++) {
                        float temp = weights.get(cells[0],cells[1],k)*(exp(q1[idx++]) - voxel.get(cells[0],cells[1],k));
                        w += weights.get(cells[0],cells[1],k);
                        float deli = del.get(cells[0], cells[1], k);
                        del.set(cells[0],cells[1],k, deli+temp);
                    }
                }
            }
       // }
        for(int i=0; i<N; i++){
            for(int j=0; j<N; j++){
                for(int k=0; k<N; k++){
                    float ti = voxel.get(i,j,k);
                    float di = del.get(i,j,k);
                    if(ti + di/w > 0.7){        //threshold is equal to 0.7 for now
                        voxel.set(i,j,k,1);
                    }else {
                        voxel.set(i, j, k, ti + di / w);
                    }
                }
            }
        }

    }
    return voxel;

}

