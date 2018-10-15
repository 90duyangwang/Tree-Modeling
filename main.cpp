#include <cstdio>
#include <opencv2/opencv.hpp>
#include <iostream>


using namespace cv;
using namespace std;


static Mat gradientX(Mat & mat, float spacing) {
    Mat grad = Mat::zeros(mat.cols,mat.rows,CV_32F);

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
    Mat grad = Mat::zeros(mat.cols,mat.rows,CV_32F);

    /*  last row */
    const int maxCols = mat.cols;
    const int maxRows = mat.rows;

    /* get gradients in each border */
    /* first row */
    Mat row = (-mat.row(0) + mat.row(1))/(float)spacing;
    row.copyTo(grad(Rect(0,0,maxCols,1)));

    row = (-mat.row(maxRows-2) + mat.row(maxRows-1))/(float)spacing;
    row.copyTo(grad(Rect(0,maxRows-1,maxCols,1)));

    /* centered elements */
    Mat centeredMat = mat(Rect(0,0,maxCols,maxRows-2));
    Mat offsetMat = mat(Rect(0,2,maxCols,maxRows-2));
    Mat resultCenteredMat = (-centeredMat + offsetMat)/(((float)spacing)*2.0);

    resultCenteredMat.copyTo(grad(Rect(0,1,maxCols, maxRows-2)));
    return grad;
}







int main(int argc, char** argv )
{

    Mat_<uchar> image = imread("./Images/troll.png", 0);
    Mat_<uchar> trimap = imread("./Images/trollTrimap.bmp", 0);
    Mat_<uchar> foreground(trimap.size(), (uchar)0);
    Mat_<uchar> background(trimap.size(), (uchar)0);
    Mat_<uchar> unknown(trimap.size(), (uchar)0);
    Mat_<uchar> fg_mask(trimap.size(), (uchar)0);
    Mat_<uchar> bg_mask(trimap.size(), (uchar)0);

    int h = trimap.rows;
    int w = trimap.cols;
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            if (trimap(y, x) == 0)
                background(y, x) = 1;
            else if (trimap(y, x) == 255)
                foreground(y, x) = 1;
            unknown(y, x) = 1^(foreground(y, x) || background(y, x));
            bg_mask = (unknown(y, x) + foreground(y, x)) * 255;
            fg_mask = (unknown(y, x) + background(y, x)) * 255;
        }
    }
    int h1 = image.rows;
    int c1 = image.cols;

    Mat_<uchar> fg_image(image.size(), (uchar)0);
    Mat_<uchar> bg_image(image.size(), (uchar)0);
    Mat_<uchar> alphaestimate(image.size(), (uchar)0);

    for(int i = 0; i < h; i++){
        for(int j = 0; j < w; j++){
            fg_image(i,j) = image(i,j) * foreground(i, j);
            bg_image(i,j) = image(i,j) * background(i, j);
            alphaestimate(i, j) = foreground(i, j) + 0.5 * unknown(i, j);
        }
    }


    Mat_<uchar> approx_fg(image.size(), (uchar)0);
    Mat_<uchar> approx_bg(image.size(), (uchar)0);

    inpaint(bg_image, bg_mask, approx_bg, 3, INPAINT_TELEA);
    inpaint(fg_image, fg_mask, approx_fg, 3, INPAINT_TELEA);

    for(int i = 0; i < h; i++){
        for(int j = 0; j < w; j++){
            approx_fg(i,j) = approx_fg(i, j) * (background(i, j) == 0 ? 1:0);
            approx_bg(i,j) = approx_bg(i, j) * (foreground(i, j) == 0 ? 1:0);

        }
    }


    // approx_diff = approx_fg - approx_bg
    // approx_diff = scipy.ndimage.filters.gaussian_filter(approx_diff, 0.9)
    Mat_<uchar> approx_diff(image.size(), (uchar)0);
    for(int i=0 ; i < h; i++){
        for (int j=0 ;j < w; j++){
            approx_diff(i, j) = approx_fg(i, j) - approx_bg(i, j);
        }
    }

    //GaussianBlur(approx_diff, approx_diff, 3 ,0.9);


    Mat dy = gradientY(image, 1.0);
    Mat dx = gradientX(image, 1.0);
    cout << dy.rows << dy.cols << endl;
    //cout<<dy.at<float>(0,0) << endl;
    //cout<<dx.at<float>(0.0) << endl;

    Mat diff_dy;
    Mat diff_dx;

    divide(dy, approx_diff, diff_dy);
    divide(dx, approx_diff, diff_dx);

    Mat d2y = gradientY(diff_dy, 1.0);
    Mat d2x = gradientX(diff_dx, 1.0);


    if ( !image.data )
    {
        printf("No image data \n");
        return -1;
    }
    namedWindow("Display Image", WINDOW_AUTOSIZE );
    imshow("Display Image", image);

    waitKey(0);
    destroyAllWindows();
    return 0;
}