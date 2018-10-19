#include <cstdio>
#include <opencv2/opencv.hpp>
#include <iostream>


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



int main(int argc, char** argv )
{

    Mat image = imread("../Images/troll.png",0);
    Mat trimap = imread("../Images/trollTrimap.bmp",0);

    int h = trimap.rows;
    int w = trimap.cols;

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

    destroyAllWindows();
    return 0;
}