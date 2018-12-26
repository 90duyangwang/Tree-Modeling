//
// Created by premsai chinthamreddy on 28/11/18.
//

#ifndef IMAGE_BASED_TREE_MODELING_GUIDEDFILTER_H
#define IMAGE_BASED_TREE_MODELING_GUIDEDFILTER_H
#include <opencv2/opencv.hpp>

class GuidedFilterImpl;

class GuidedFilter
{
public:
    GuidedFilter(const cv::Mat &I, int r, double eps);
    ~GuidedFilter();

    cv::Mat filter(const cv::Mat &p, int depth = -1) const;

private:
    GuidedFilterImpl *impl_;
};

cv::Mat guidedFilter(const cv::Mat &I, const cv::Mat &p, int r, double eps, int depth = -1);

#endif //IMAGE_BASED_TREE_MODELING_GUIDEDFILTER_H
