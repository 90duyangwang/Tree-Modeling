#include <cstdio>
#include <opencv2/opencv.hpp>
#include <iostream>


using namespace cv;
using namespace std;


static Mat gradientX(Mat & mat, float spacing) {
    Mat grad = Mat::zeros(mat.rows,mat.cols,CV_32F);

    /*  last row */
    int maxCols = mat.cols;
    int maxRows = mat.rows;

    /* get gradients in each border */
    /* first row */
    Mat col = (-mat.col(0) + mat.col(1))/(float)spacing;
    col.copyTo(grad(Rect(0,0,1,maxRows)));

    col = (-mat.col(maxCols-2) + mat.col(maxCols-1))/(float)spacing;
    col.copyTo(grad(Rect(maxCols-1,0,1,maxRows)));

    /* centered elements */
    Mat centeredMat = mat(Rect(0,0,maxCols-2,maxRows));
    Mat offsetMat = mat(Rect(2,0,maxCols-2,maxRows));
    Mat resultCenteredMat = (-centeredMat + offsetMat)/(((float)spacing)*2.0);

    resultCenteredMat.copyTo(grad(Rect(1,0,maxCols-2, maxRows)));
    return grad;
}

/// Internal method to get numerical gradient for y components.
/// @param[in] mat Specify input matrix.
/// @param[in] spacing Specify input space.
static Mat gradientY(Mat & mat, float spacing) {
    Mat grad = Mat::zeros(mat.rows, mat.cols, CV_32F);

    /*  last row */
    const int maxCols = mat.cols;
    const int maxRows = mat.rows;

    /* get gradients in each border */
    /* first row */
    Mat row = (-mat.row(0) + mat.row(1))/(float)spacing;
    Rect rect(0, 0, maxCols, 1);
    Rect rect2(0, maxRows-1, maxCols, 1);

    row.copyTo(grad(rect));
    row = (-mat.row(maxRows-2) + mat.row(maxRows-1))/(float)spacing;
    row.copyTo(grad(rect2));

    /* centered elements */
    Mat centeredMat = mat(Rect(0,0,maxCols,maxRows-2));
    Mat offsetMat = mat(Rect(0,2,maxCols,maxRows-2));
    Mat resultCenteredMat = (-centeredMat + offsetMat)/(((float)spacing)*2.0);

    resultCenteredMat.copyTo(grad(Rect(0,1,maxCols, maxRows-2)));

    return grad;
}



int main(int argc, char** argv )
{

    Mat image = imread("../Images/troll.png", 0);
    Mat trimap = imread("../Images/trollTrimap.bmp", 0);

    int h = trimap.rows;
    int w = trimap.cols;

    Mat foreground = Mat::ones(trimap.size(), CV_8U);
    Mat background = Mat::zeros(trimap.size(), CV_8U);
    Mat unknown(trimap.size(), CV_8U);

    Mat fg_mask(trimap.size(), CV_8U);
    Mat bg_mask(trimap.size(), CV_8U);

    bitwise_xor(background, trimap, background);
    bitwise_and(foreground, trimap, foreground);

    // Refactoring code
    bitwise_or(foreground, background, unknown);
    bitwise_xor(unknown, Mat::ones(h, w, CV_8U), unknown);

    bg_mask = (unknown + foreground) * 255;
    fg_mask = (unknown + background) * 255;

    Mat fg_image = image.mul(foreground);
    Mat bg_image = image.mul(background);

    namedWindow("Display Image", WINDOW_AUTOSIZE );
    imshow("Display Image", fg_image);
    imshow("Image Background", bg_image);

    waitKey(0);

    Mat alphaestimate = foreground + 0.5 * unknown;
    Mat approx_bg;
    Mat approx_fg;

    cout<<image.size() << "Image size" << endl;
    inpaint(bg_image, bg_mask, approx_bg, 3, INPAINT_TELEA);
    inpaint(fg_image, fg_mask, approx_fg, 3, INPAINT_TELEA);


    namedWindow("Display Image", WINDOW_AUTOSIZE );

    imshow("Display Image", approx_fg);
    waitKey(0);
    imshow("Display Image", approx_bg);
    waitKey(0);

    Mat backnot;
    bitwise_not(background, backnot);
    Mat forenot;
    bitwise_not(foreground, forenot);

    approx_fg = approx_fg.mul(backnot);
    approx_bg = approx_bg.mul(forenot);

    Mat fg_img;
    approx_fg.convertTo(fg_img, CV_32FC1);
    Mat bg_img;
    approx_bg.convertTo(bg_img, CV_32FC1);
    Mat approx_diff = fg_img - bg_img;


    Size sz(3,3);

    GaussianBlur(approx_diff, approx_diff, sz ,0.9);
    Mat image2 = imread("../Images/troll.png", 0);

    Mat dy = gradientY(image2, 1.0);
    Mat dx = gradientX(image2, 1.0);

    Mat diff_dy;
    Mat diff_dx;

    divide(dy, approx_diff, diff_dy);
    divide(dx, approx_diff, diff_dx);

    Mat d2y = gradientY(diff_dy, 1.0);
    Mat d2x = gradientX(diff_dx, 1.0);
    Mat b = d2y + d2x;



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
    while (n<50 && sum(diff_result)[0] > threshold) {
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

    namedWindow("Display Image", WINDOW_AUTOSIZE );
    imshow("Display Image", alpha);

    waitKey(0);
    destroyAllWindows();
    return 0;
}