#include <cstdio>
#include <opencv2/opencv.hpp>
#include <iostream>
#include "vec3d.h"
#include "voxel_reconstruction.h"
#include "globalmatting.h"
#include "guidedfilter.h"
#include "glm/vec3.hpp"
#include "PixelGraph.h"

using namespace cv;
using namespace std;


Mat gradientX(Mat & mat, float spacing) {
    Mat grad = Mat::zeros(mat.rows,mat.cols,CV_32F);

    int maxCols = mat.cols;
    int maxRows = mat.rows;

    Mat col = (-mat.col(0) + mat.col(1))/(float)spacing;
    col.copyTo(grad(Rect(0,0,1,maxRows)));

    col = (-mat.col(maxCols-2) + mat.col(maxCols-1))/(float)spacing;
    col.copyTo(grad(Rect(maxCols-1,0,1,maxRows)));

    Mat centeredMat = mat(Rect(0,0,maxCols-2,maxRows));
    Mat offsetMat = mat(Rect(2,0,maxCols-2,maxRows));
    Mat resultCenteredMat = (-centeredMat + offsetMat)/(((float)spacing)*2.0);

    resultCenteredMat.copyTo(grad(Rect(1,0,maxCols-2, maxRows)));
    return grad;
}

Mat gradientY(Mat & mat, float spacing) {
    Mat grad = Mat::zeros(mat.rows, mat.cols, CV_32F);
//    Mat grad = res;

    const int maxCols = mat.cols;
    const int maxRows = mat.rows;

    Mat row = (-mat.row(0) + mat.row(1))/(float)spacing;
    Rect rect(0, 0, maxCols, 1);
    Rect rect2(0, maxRows-1, maxCols, 1);

    row.copyTo(grad(rect));
    row = (-mat.row(maxRows-2) + mat.row(maxRows-1))/(float)spacing;
    row.copyTo(grad(rect2));

    Mat centeredMat = mat(Rect(0,0,maxCols,maxRows-2));
    Mat offsetMat = mat(Rect(0,2,maxCols,maxRows-2));
    Mat resultCenteredMat = (-centeredMat + offsetMat)/(((float)spacing)*2.0);

    resultCenteredMat.copyTo(grad(Rect(0,1,maxCols, maxRows-2)));

    return grad;
}

Mat alphaEstimation(string &inputImage, string &bitmapfile) {
    Mat image = imread(inputImage,0);
    Mat trimap = imread(bitmapfile,0);

    int h = trimap.rows;
    int w = trimap.cols;


    if(h==0 || w == 0 ){
        cout << " Files not read properly"<<endl;
        exit(0);
    }


    Mat foreground = (trimap == 255);
    Mat background = (trimap == 0);
    Mat unknown = (trimap == 128);


    Mat bg_mask;
    bitwise_or(unknown, foreground, bg_mask);
    Mat fg_mask;
    bitwise_or(unknown, background, fg_mask);


    Mat fg_image;
    Mat bg_image;
    image.copyTo(fg_image, foreground);
    image.copyTo(bg_image, background);

    Mat alphaestimate = foreground + 0.5 * unknown;
    Mat approx_bg;
    Mat approx_fg;

    inpaint(bg_image, bg_mask, approx_bg, 3, INPAINT_TELEA);
    inpaint(fg_image, fg_mask, approx_fg, 3, INPAINT_TELEA);

    Mat backnot;
    bitwise_not(background, backnot);
    Mat forenot;
    bitwise_not(foreground, forenot);


    Mat approx_fg2, approx_bg2;

    approx_fg.copyTo(approx_fg2, backnot);
    approx_bg.copyTo(approx_bg2, forenot);

    Mat approx_fg3;
    approx_fg.convertTo(approx_fg3, CV_32FC1);
    Mat approx_bg3;
    approx_bg.convertTo(approx_bg3, CV_32FC1);

    //Mat approx_diff = fg_img - bg_img;

    Mat approx_diff = approx_fg3 - approx_bg3;

    Size sz(7,7);
    Mat guass_res;
    GaussianBlur(approx_diff, guass_res, sz ,0.9);

    //Mat dy = Mat::zeros(image.rows, image.cols, CV_32FC1);

    Mat dy = gradientY(image, 1.0);
    Mat dx = gradientX(image, 1.0);

//    cout << dy << endl;

    Mat diff_dy;
    Mat diff_dx;

    divide(dy, guass_res, diff_dy);
    divide(dx, guass_res, diff_dx);

    cout << guass_res.row(0) << endl;
    cout << dy.row(0)<<endl;
    cout << diff_dy.row(0) << endl;

    Mat d2y = gradientY(diff_dy, 1.0);
    Mat d2x = gradientX(diff_dx, 1.0);
    Mat b = d2y + d2x;

    cout << d2y.row(0) <<endl;

// Alpha calculation
    Mat alphaest;
    alphaestimate.convertTo(alphaest, CV_32F);
    Mat alphaNew;
    alphaest.copyTo(alphaNew);
    Mat alphaOld = Mat::zeros(alphaNew.rows, alphaNew.cols, CV_32F);
    float threshold = 0.1;
    int n = 1;
    Mat diff_result;
    absdiff(alphaNew, alphaOld, diff_result);
    while (n<100 && sum(diff_result)[0] > threshold) {
        alphaNew.copyTo(alphaOld);
        for(int i=1; i<h-1; i++) {
            for(int j=1; j<w-1; j++) {
                if(unknown.at<unsigned char>(i,j) != 0) {
                    alphaNew.at<float>(i,j) = (1/4.0)  * (alphaNew.at<float>(i-1 ,j) + alphaNew.at<float>(i,j-1)
                                                          + alphaOld.at<float>(i, j+1) + alphaOld.at<float>(i+1,j) - b.at<float>(i,j));
                }
            }
        }
        n = n+1;
    }
    Mat alpha = min(max(alphaNew,0),1);
    cout << "Completed Alpha estimate" << endl;

    namedWindow("Image Alpha", WINDOW_AUTOSIZE);
    imshow("Image Alpha", alpha);
    waitKey(0);
    destroyAllWindows();
    return alpha;
}


vector<vector<float>> create_vec(Mat &alpha) {
    vector<vector<float>> res;
    vector<float> temp;
    Mat alpha2;
    alpha.convertTo(alpha2, CV_32FC1);
    for(int i=0; i< alpha.rows ; i++){
        for(int j=0; j<alpha.cols; j++){
            float temp1 = alpha2.at<float>(i,j);
            temp.push_back(temp1/255.0);
        }
        res.push_back(temp);
        temp.clear();
    }
    return res;
}

Mat create_Mat(vector<vector<float>> &alpha){
    cv::Mat matAlpha(alpha.size(), alpha.at(0).size(), CV_32FC1);
    for(int i=0; i<matAlpha.rows; ++i)
        for(int j=0; j<matAlpha.cols; ++j)
            matAlpha.at<float>(i, j) = alpha.at(i).at(j);

    return matAlpha;
}


int main(int argc, char** argv )
{
    string imagefile = "./Images/troll.png";
    string bitmapfile = "./Images/trollTrimap.bmp";
    Mat image = imread(imagefile, 1);
    Mat trimap = imread(bitmapfile,0);
    expansionOfKnownRegions(image, trimap, 9);
    Mat foreground, alpha;
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
    namedWindow("Image Alpha", WINDOW_AUTOSIZE);
    imshow("Image Alpha", alpha);
    waitKey(0);
    destroyAllWindows();




    cv::imwrite("GT04-alpha.png", alpha);

//    Mat alpha = alphaEstimation(imagefile, bitmapfile);
    vector<vector<float>> alpha_img = create_vec(alpha);
    std::cout << alpha;

    int w = image.cols;
    int h = image.rows;
    PixelGraph pg((int)h/10.0, (int)w/10.0);
    bool attractor = false;
    int attx = -1, atty = -1;
    for(int i=0,m=0; i<h ;i+=10,m++) {
        for(int j=0,n=0; j<w; j+=10,n++) {
            attractor = false;
            attx = -1;
            atty = -1;
            for(int x=i; x<i+10; x++){
                for(int y=j; y<j+10; y++){
                    const cv::Vec3b &intensity = image.at<Vec3b>(x, y);
                    uchar red = intensity.val[2];
                    if(red == 255){
                        attractor = true;
                        attx = x;
                        atty = y;
                    }
                }
            }
            PixelNode pnode(-1,-1);
            if(attractor) {
                pnode.setCoordinates(attx, atty);
                pnode.setAttracPoint(true);
            }else{
                pnode.setCoordinates(i+5, j+5);
                pnode.setDirecPoint(true);
            }
            pg.set(m, n, &pnode);
        }
    }


    vec3d voxel_alpha = alpha_calculation(alpha.rows, alpha.cols, alpha_img);
    cout << "Voxels generated" << endl;
    glm::vec3 vectemp(-1.0f,0.0f,0.0f);
    vector<vector<float>> voxel_test;

    for(int i=0; i<1; i++){
//        char tmp = 'a'+i;
////        string filename = "./Images/layer_" + tmp;
////        freopen(filename.c_str(), "w", stdout);
        for(int j=0; j<25; j++){

            vector<float> temp_test;
            for(int k=0; k<25; k++){
                temp_test.push_back(voxel_alpha.get(j,k,i));
            }
            voxel_test.push_back(temp_test);
            temp_test.clear();
        }
////        fclose(stdout);
    }
    Mat m1 = create_Mat(voxel_test);
    cv::imwrite("voxel_alpha.png", m1);








    return 0;
}