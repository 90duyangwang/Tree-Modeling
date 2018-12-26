#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>


#include "globalmatting.h"
#include "guidedfilter.h"


#include "PixelNode.h"
#include "PixelGraph.h"
#define UPI 3.14159265359
#define display_counter_MAX 10000


#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <iterator>
#include <random>
#include <string>
#include <sstream>
#include <vector>


const int VOXEL_DIM = 25;
const int VOXEL_SIZE = VOXEL_DIM*VOXEL_DIM*VOXEL_DIM;
const int VOXEL_SLICE = VOXEL_DIM*VOXEL_DIM;

const int IMG_WIDTH = 1280;
const int IMG_HEIGHT = 960;

const int OUTSIDE = 0;
const int no_images = 2;


struct voxel {
    float xpos;
    float ypos;
    float zpos;
    float res;
    float value;
};


struct coord {
    int x;
    int y;
};

struct startParams {
    float startX;
    float startY;
    float startZ;
    float voxelWidth;
    float voxelHeight;
    float voxelDepth;
};

struct camera {
    cv::Mat Image;
    cv::Mat P;
    cv::Mat K;
    cv::Mat R;
    cv::Mat t;
    cv::Mat silhouette;
};

struct voxel_coords {
    int i;
    int j;
    int k;
};

//Store the vector field information
typedef struct p_vector{
    double xcm;
    double ycm;
    double vx;
    double vy;
}p_vector;

//Dimensions of the image the data came from
unsigned int xsize = 300;
unsigned int ysize = 300;

//Scale factor used to scale the arrow size
const double scale = 0.05l;//2.75f;

//Array to store the vector field information structure
struct p_vector *par;


coord project(camera cam, voxel v) {

    coord im;

    /* project voxel into camera image coords */
    float z =   cam.P.at<float>(2, 0) * v.xpos +
                cam.P.at<float>(2, 1) * v.ypos +
                cam.P.at<float>(2, 2) * v.zpos +
                cam.P.at<float>(2, 3);

    im.y =    (cam.P.at<float>(1, 0) * v.xpos +
               cam.P.at<float>(1, 1) * v.ypos +
               cam.P.at<float>(1, 2) * v.zpos +
               cam.P.at<float>(1, 3)) / z;

    im.x =    (cam.P.at<float>(0, 0) * v.xpos +
               cam.P.at<float>(0, 1) * v.ypos +
               cam.P.at<float>(0, 2) * v.zpos +
               cam.P.at<float>(0, 3)) / z;

    return im;
}


void intialize_voxel_densities (float fArray[], startParams params, camera cam) {
    cv::Mat image = cam.Image;
    for (int i=0; i<VOXEL_DIM; i++) {
        for (int j=0; j<VOXEL_DIM; j++) {
            for (int k=0; k<VOXEL_DIM; k++) {

                /* calc voxel position inside camera view frustum */
                voxel v;
                v.xpos = params.startX + i * params.voxelWidth;
                v.ypos = params.startY + j * params.voxelHeight;
                v.zpos = params.startZ + k * params.voxelDepth;
                v.value = 1.0f;

                coord im = project(cam, v);

                /* test if projected voxel is within image coords */
                if (im.x > 0 && im.y > 0 && im.x < IMG_WIDTH && im.y < IMG_HEIGHT) {
                    fArray[i*VOXEL_SLICE+j*VOXEL_DIM+k] += image.at<float>(im.x, im.y);
                }

            }
        }
    }
}

std::vector<float> project_q_onto_plane(std::vector<float> &q, float qp, int n) {
    // Equation of plane using Normal N(1,1,1,...,1) and intercept qp - n dimensions
    // using the equation of the plane in n-dimension find the projection of q on that plane
    // t = ad-ax + be -by + cf -cz /(a2 + b2 +c2)
    // N = (a,b,c)
    // qp = (qp,0,0) = (d,e,f)
    // q = (x,y,z)
    float sqsum = 0.0;
    std::vector<float> N(n,1);
    for(int i=0; i<n; i++){
        sqsum += N[i]*N[i];
    }
    float neg_sum = 0.0;
    for(int i=0; i<n; i++){
        neg_sum += N[i]*q[i];
    }
    float t = (((qp - neg_sum)*1.0)/sqsum);
    std::vector<float> res;
    for(int i=0; i<n; i++) {
        res.push_back(q[i] + t*N[i]);
    }

    return res;

}


std::vector<voxel_coords> list_cells(int x, int y, startParams params, camera cam) {

    std::vector<voxel_coords> res;
    for (int i = 0; i < VOXEL_DIM; i++) {
        for (int j = 0; j < VOXEL_DIM; j++) {
            for (int k = 0; k < VOXEL_DIM; k++) {
                voxel v;
                v.xpos = params.startX + i * params.voxelWidth;
                v.ypos = params.startY + j * params.voxelHeight;
                v.zpos = params.startZ + k * params.voxelDepth;
                v.value = 1.0f;

                coord im = project(cam, v);

                if (im.x > 0 && im.y > 0 && im.x < IMG_WIDTH && im.y < IMG_HEIGHT) {
                    if(im.x == x && im.y == y){
                        voxel_coords vpos;
                        vpos.i = i;
                        vpos.j = j;
                        vpos.k = k;
                        res.push_back(vpos);
                    }
                }

            }
        }
    }

    return res;

}

//
//void carve(float fArray[], startParams params, camera cam) {
//    cv::Mat silhouette, distImage;
//    cv::Canny(cam.silhouette, silhouette, 0, 255);
//    cv::bitwise_not(silhouette, silhouette);
//    cv::distanceTransform(silhouette, distImage, CV_DIST_L2, 3);
//
//    for (int i=0; i<VOXEL_DIM; i++) {
//        for (int j=0; j<VOXEL_DIM; j++) {
//            for (int k=0; k<VOXEL_DIM; k++) {
//
//                /* calc voxel position inside camera view frustum */
//                voxel v;
//                v.xpos = params.startX + i * params.voxelWidth;
//                v.ypos = params.startY + j * params.voxelHeight;
//                v.zpos = params.startZ + k * params.voxelDepth;
//                v.value = 1.0f;
//
//                coord im = project(cam, v);
//                float dist = -1.0f;
//
//                /* test if projected voxel is within image coords */
//                if (im.x > 0 && im.y > 0 && im.x < IMG_WIDTH && im.y < IMG_HEIGHT) {
//                    dist = distImage.at<float>(im.y, im.x);
//                    if (cam.silhouette.at<uchar>(im.y, im.x) == OUTSIDE) {
//                        dist *= -1.0f;
//                    }
//                }
//
//                if (dist < fArray[i*VOXEL_SLICE+j*VOXEL_DIM+k]) {
//                    fArray[i*VOXEL_SLICE+j*VOXEL_DIM+k] = dist;
//                }
//
//            }
//        }
//    }
//
//}

void alpha_calculation(float fArray[], startParams params, std::vector<camera> cams, float del[]) {
    float threshold = 0.675;
//    int lst_imgs = no_images;
    int img_idx = 0;
    for (int img_idx = 0; img_idx < no_images; img_idx++) {
        camera cam = cams.at(img_idx);
        std::vector<float> q;
        for (int i = 0; i < IMG_HEIGHT; i++) {
            for (int j = 0; j < IMG_WIDTH; j++) {
                
                const cv::Vec3b &intensity = cam.Image.at<cv::Vec3b>(i, j);
                uchar red = intensity.val[2];
                uchar green = intensity.val[1];
                uchar black = intensity.val[0];
                int redc = (int)red;
                int greenc = (int)greenc;
                int blackc = (int)blackc;
                if(redc < 100)
                
                std::vector<voxel_coords> cells = list_cells(i, j, params, cam);
                for (int k = 0; k < cells.size(); k++) {
                    q.push_back(log(fArray[cells[k].i * VOXEL_SLICE + cells[k].j * VOXEL_DIM + cells[k].k]));
                }
                float qp = log(1 - cam.Image.at<float>(i, j));
                std::vector<float> q1 = project_q_onto_plane(q, qp, cells.size());
                for (int k = 0; k < cells.size(); k++) {
                    del[cells[k].i * VOXEL_SLICE + cells[k].j * VOXEL_DIM + cells[k].k] +=
                            exp(q1[k]) - fArray[cells[k].i * VOXEL_SLICE + cells[k].j * VOXEL_DIM + cells[k].k];
                }

            }

        }
    }
    for (int i=0; i<VOXEL_DIM; i++) {
        for (int j = 0; j < VOXEL_DIM; j++) {
            for (int k = 0; k < VOXEL_DIM; k++) {
                fArray[i * VOXEL_SLICE + j * VOXEL_DIM + k] += del[i * VOXEL_SLICE + j * VOXEL_DIM + k];
                if(fArray[i * VOXEL_SLICE + j * VOXEL_DIM + k] > threshold) {
                    fArray[i * VOXEL_SLICE + j * VOXEL_DIM + k] = 1;
                }
            }
        }
    }

}


void pre_processing(){
    std::vector<camera> cams;
    cv::FileStorage fs("/Users/premsai/Desktop/vector_field_2/assets/viff.xml", cv::FileStorage::READ);
    for (int i=0; (i==0 || i==10)&& i<12; i++) {
        std::stringstream fimg;
        fimg << "merged_" << i << ".jpg";
        cv::Mat img = cv::imread(fimg.str());

        cv::Mat silhouette;
        cv::cvtColor(img, silhouette, CV_BGR2HSV);
        cv::inRange(silhouette, cv::Scalar(0,0,30), cv::Scalar(255,255,255), silhouette);

        std::stringstream img_mat;
        img_mat << "viff" << i << "_matrix";
        cv::Mat P;
        fs[img_mat.str()] >> P;

        cv::Mat K, R, t;
        cv::decomposeProjectionMatrix(P, K, R, t);
        K = cv::Mat::eye(3, 3, CV_32FC1);
        K.at<float>(0,0) = 1680.263141;  /* fx */
        K.at<float>(1,1) = 1676.120436;  /* fy */
        K.at<float>(0,2) = 621.5917832;  /* cx */
        K.at<float>(1,2) = 467.7223334;  /* cy */

        camera c;
        c.Image = img;
        c.P = P;
        c.K = K;
        c.R = R;
        c.t = t;
        c.silhouette = silhouette;

        cams.push_back(c);

    }

    float xmin = -6.21639, ymin = -10.2796, zmin = -14.0349;
    float xmax = 7.62138, ymax = 12.1731, zmax = 12.5358;

    float bbwidth = std::abs(xmax-xmin)*1.15;
    float bbheight = std::abs(ymax-ymin)*1.15;
    float bbdepth = std::abs(zmax-zmin)*1.05;

    startParams params;
    params.startX = xmin-std::abs(xmax-xmin)*0.15;
    params.startY = ymin-std::abs(ymax-ymin)*0.15;
    params.startZ = 0.0f;
    params.voxelWidth = bbwidth/VOXEL_DIM;
    params.voxelHeight = bbheight/VOXEL_DIM;
    params.voxelDepth = bbdepth/VOXEL_DIM;

    /* 3 dimensional voxel grid */
    float *fArray = new float[VOXEL_SIZE];
    std::fill_n(fArray, VOXEL_SIZE, 0.0f);

    /* 3 dimensional voxel grid */
    float *del = new float[VOXEL_SIZE];
    std::fill_n(del, VOXEL_SIZE, 0.0f);

    for (int i=0; i<no_images; i++) {
        intialize_voxel_densities(fArray, params, cams.at(i));
    }

    for (int i=0; i<VOXEL_DIM; i++) {
        for (int j = 0; j < VOXEL_DIM; j++) {
            for (int k = 0; k < VOXEL_DIM; k++) {
                fArray[i * VOXEL_SLICE + j * VOXEL_DIM + k] /= no_images;
                fArray[i * VOXEL_SLICE + j * VOXEL_DIM + k] = 1 - fArray[i * VOXEL_SLICE + j * VOXEL_DIM + k];
            }
        }
    }

    /* carving model for every given camera image */
    int convergence_iterations = 0
    while(convergence_iterations <= 3) {
        std::fill_n(del, VOXEL_SIZE, 0.0f);
        alpha_calculation(fArray, params, cams, del);
        convergence_iterations++;
    }

}

std::vector<std::vector<float>> image_alpha(std::string imagefile, std::string bitmapfile) {
//    std::string imagefile = "./Images/troll.png";
//    std::string bitmapfile = "./Images/trollTrimap.bmp";
    cv::Mat image = cv::imread(imagefile, 1);
    cv::Mat trimap = cv::imread(bitmapfile,0);
    expansionOfKnownRegions(image, trimap, 9);
    cv::Mat foreground, alpha;
    globalMatting(image, trimap, foreground, alpha);

    alpha = guidedFilter(image, alpha, 10, 1e-5);
    for (int x = 0; x < trimap.cols; ++x)
        for (int y = 0; y < trimap.rows; ++y)
        {
            if (trimap.at<uchar>(y, x) == 0)
                alpha.at<uchar>(y, x) = 0;
            else if (trimap.at<uchar>(y, x) == 255)
                alpha.at<uchar>(y, x) = 255;
        }
    cv::namedWindow("Image Alpha", cv::WINDOW_AUTOSIZE);
    cv::imshow("Image Alpha", alpha);
    cv::waitKey(0);
    cv::destroyAllWindows();

    /* Normalize the alpha mask to 0..1*/
    std::vector<std::vector<float>> img_alpha;
    std::vector<float> temp;
    cv::Mat alpha2;
    alpha.convertTo(alpha2, CV_32FC1);
    for(int i=0; i< alpha.rows ; i++){
        for(int j=0; j<alpha.cols; j++){
            float temp1 = alpha2.at<float>(i,j);
            temp.push_back(temp1/255.0);
        }
        img_alpha.push_back(temp);
        temp.clear();
    }

    return img_alpha;

}


// Generate vector field from attractor graph

void generate_vector_fields(std::string img_filename) {
    
    char imfile[100] = "/Users/premsai/Desktop/vector_field_2/assets/merged_attractor_0\0";
    cv::Mat image = cv::imread(imfile,1);
    int w = image.cols;
    int h = image.rows;
    
    xsize = h;
    ysize = w;
    
    PixelGraph pg((int)h/50.0 + 1, (int)w/50.0 + 1);
    bool attractor = false;
    int attx = -1, atty = -1;
    int attraccount = 0;
    int direccount = 0;
    PixelNode *pnode;
    for(int i=0,m=0; i<h ;i+=50,m++) {
        for(int j=0,n=0; j<w; j+=50,n++) {
            attractor = false;
            attx = -1;
            atty = -1;
            
            for(int x=i; x<i+50 && x<h; x++){
                for(int y=j; y<j+50 && y<w; y++){
                    const cv::Vec3b &intensity = image.at<cv::Vec3b>(x, y);
                    uchar red = intensity.val[2];
                    uchar green = intensity.val[1];
                    uchar black = intensity.val[0];
                    int redc = (int)red;
                    int greenc = (int)greenc;
                    int blackc = (int)blackc;
                    // The color of the attrctor graph drawn manually
                    if(redc == 237 && greenc == 28 && black == 36){
                        
                        attractor = true;
                        attx = x;
                        atty = y;
                    }else{
                        
                        attractor = false;
                        attx = x;
                        atty = y;
                    }
                }
            }
            pnode = new PixelNode(-1,-1);
            if(attractor) {
                attraccount++;
                pnode->setCoordinates(attx, atty);
                pnode->setAttracPoint(true);
            }else{
                direccount++;
                pnode->setCoordinates(i+5, j+5);
                pnode->setDirecPoint(true);
            }
            pnode->setisVisited(false);
            pg.set(m, n, pnode);
        }
    }

    int height = pg.height;
    int width = pg.width;
    unsigned int q = 0;
    pnum = direccount;
    par = (struct p_vector*)malloc(pnum * sizeof(struct p_vector));
    for(int i=0; i<height; i++) {
        for(int j=0; j<width; j++) {
            if(!pg.get(i, j)->isAttracPoint) {
                std::pair<int, int> head = pg.getNearestNeighbour(i, j);
                
                for(int i1=0; i1<height; i1++){
                    for(int j1=0; j1<width; j1++){
                        pg.get(i1, j1)->setisVisited(false);
                    }
                }
                
                if(head.first != -1 && head.second != -1){
                    
                    float x1 = pg.get(head.first, head.second)->x;
                    float y1 = pg.get(head.first, head.second)->y;
                    
                    float x2 = i;
                    float y2 = j;
                    
                    float vx = x1 - x2;
                    float vy = y1 - y2;
                    float s= 1.0f / sqrt(vx * vx + vy * vy);
                    vx *= s;
                    vy *= s;
                    pg.get(i, j)->setVectors(vx, vy);
                    
//                    par[q].xcm = (x1+x2)/2;
//                    par[q].ycm = (y1+y2)/2;
//                    
//                    par[q].vx  = vx;
//                    par[q].vy  = vy;
                    
                    q++;
                }
            }
        }
    }
}

// Fetch pixel node from image coordinates
//
PixelNode* get_pixel_vector(int imgx, int imgy, PixelGraph *pg) {
    int x = imgx/50.0;
    int y = imgy/50.0;
    return pg.get(x, y);
}





// Particle Systems
// Voxels are filled
// Generate particles at random positions

namespace utils {
    static const float M_PI = 3.1415926535897932384626433832795f;
    
    template <typename T> T elapsed_time() {
        using namespace std::chrono;
        static time_point<steady_clock> start = steady_clock::now();
        return duration_cast<duration<T>>(steady_clock::now() - start).count();
    }
    
    float gen_random(float x0, float y0, float z0, float x1, float y1, float z1) {
        // Get random positions from the givel voxel positions
        // (x0,y0,z0) and (x1, y1, z1) are the two vertices of the cube(voxel)
        // Front top left and back right bottom
        // The random positions must be inside the voxel boundaries
    }
    int count_non_transparent_voxels() {
        // Take filled voxel grid as the input to estimate the number of
        // non-transparent voxels to distribute the particles.
    }
    
    void generate_all_random_positions(int nparts) {
        // Generate all the random positions for the particles
        // Take input as the number of particles
        // store them in a file or return them as a data structure
    }
    
    template float elapsed_time();
}






namespace graphics {
    struct Particle {
        Particle(float x, float y, float z, float scale) :
        px(x),
        py(y),
        pz(z),
        mass(scale),
        isAlive(true)
        {
            
        }
        
        float px, py, pz;
        float mass;
        bool isAlive;
        
        // Keep trace of all the previous positions
        // Update the particle position
        std::vector<maths::vec3f> trace;

    };
    
    
    class ParticleSystem {
    public:
        
        
        int num_particles = 0;
        std::vector<Particle> particles;
        ParticleSystem(int num_particles = 10) : particles()
        {
            for (int i = 0; i < num_particles; ++i) {
                // Get position (x,y,z) for each particle
                // Pass the particle position while initializing the particle
                
                float x,y,z;
                particles.push_back(Particle(x, y, z));
            }
        }
        
        void init_particle_system();
//        void draw_particle_system();
        
        
        
        float particle_distance(Particle &P1, Particle &P2) {
            float dist_sq = 0;
            dist_sq += (P1.px - P2.px)*(P1.px - P2.px);
            dist_sq += (P1.py - P2.py)*(P1.py - P2.py);
            dist_sq += (P1.pz - P2.pz)*(P1.pz - P2.pz);
            
            return sqrtf(dist_sq);
            
            
        }
        
        
        // Transfer the mass to the first particle
        // Transfer the traces to the first particle
        // Increase the mass of the first particle by mass of second particle
        
        void destroy_particle(Particle P1, Particle P2) {
            P1.mass += P2.mass;
            P1.trace.insert(P1.trace.end(), P2.trace.begin, P2.trace.end);
            P2.isAlive = false;
        }
        

        
        
        // Check the distance between every pair of particles
        // If the distance is below than the threshold distance
        // Merge these two particles
        // which means destroy one particle and add the trace and mass to another particle
        // Use destroy_particle function for destroyiong the
        void merge_particles() {
            for(int i=0; i<num_particles; i++) {
                for(int j=i+1; j<num_particles; j++) {
                    if(particle_distance(particles[i], particles[j]) < threshold) {
                        destroy_particle(particles[i], particles[j]);
                    }
                }
            }
            return;
        }
        
        // Project each particle onto the image
        // Determine the force direction from the vector field
        // Use a unit force in the given direction on the particle
        // Each particle has a unit mass (Assumption)
        // Update the position of all the particles
        // And after updating the positions call the merge_particles function
        
        void update_particles(){
            
            float vx = 1.0;
            float vy = 1.0;
            float vz = 1.0;
            
            for(int i=0; i<num_particles; i++){
                if(particles[i].isAlive) {
                    voxel v;
                    v.xpos = particles[i].px;
                    v.ypos = particles[i].py;
                    v.zpos = particles[i].pz;
                    v.value = 1.0f;
                    
                    coord im = project(cam, v);
                    
                    // Get the vector direction pixelnode p
                    // p.vx p.vy
                    // Todo
                    PixelNode* p = get_pixel_vector(im.x, im.y, pg);
                    float vx = p->vx;
                    float vy = p->vy;
                    // Apply the force in the vx.i + vy.j direction
                    
                    
                    float xpos, ypos, zpos;
                    
                    float accx = fx * vx;
                    float accy = fy * vy;
                    float accz = fz * vz;
                    
                    float velx = 1 * vx;
                    float vely = 1 * vy;
                    float velz = 1 * vz;
                    
                    // One second as time interval
                    float distx = velx * 1 + 0.5 * accx * 1 * 1;
                    float disty = vely * 1 + 0.5 * accy * 1 * 1;
                    float distz = velz * 1 + 0.5 * accz * 1 * 1;

                    particles[i].trace(maths::vec3f(xpos, ypos, zpos));
                    
                    particles[i].px = xpos;
                    particles[i].py = ypos;
                    particles[i].pz = zpos;
                    
                }
            }
            
            merge_particles();
        }
    };
}


// Once we have the particle positions after few iterations
// Store the number of particles and their positions
// Use the number of the particles to determine the thickness of
// each tree segment.
// Use circular discs with appropriate radii to create the tree


// Use line segments to draw the branches
namepace rendering {
    
    
class tree_model {
    public:
    std::vetcor<vec3f> branch_pts;
    tree_model(vector<math::vec3f> particle_traces): branch_pts(particle_traces) {
        
    }
    
    // draw all the traces with staright lines

};
    
    
    
    
    
    
    
    
    
    
}

int main() {
    
    // Call the image alpha function to get the image mask
    
    // Call the voxel grid generation function
    
    // After creating the voxels initialize particle system
    
    // Get the particle trace from the particle system
    
    //
    
    

    return 0;
}
