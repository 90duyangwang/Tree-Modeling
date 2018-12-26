//
// Created by premsai chinthamreddy on 28/11/18.
//

#ifndef IMAGE_BASED_TREE_MODELING_GLOBALMATTING_H
#define IMAGE_BASED_TREE_MODELING_GLOBALMATTING_H

#include <opencv2/opencv.hpp>

void expansionOfKnownRegions(cv::InputArray img, cv::InputOutputArray trimap, int niter = 9);
void globalMatting(cv::InputArray image, cv::InputArray trimap, cv::OutputArray foreground, cv::OutputArray alpha, cv::OutputArray conf = cv::noArray());

#endif //IMAGE_BASED_TREE_MODELING_GLOBALMATTING_H
