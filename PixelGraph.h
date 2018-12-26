//
// Created by premsai chinthamreddy on 03/11/18.
//

#ifndef IMAGEMATTING_PIXELGRAPH_H
#define IMAGEMATTING_PIXELGRAPH_H

#include "PixelNode.h"
#include <vector>
#include <utility>


class PixelGraph {
public:
    std::vector<std::vector<PixelNode*>> node_matrix;
    int height;
    int width;

    PixelGraph(int h, int w);
    PixelNode* get(int i, int j);
    void set(int i, int j, PixelNode* p);
    PixelNode* leftEdge(int i, int j);
    PixelNode* rightEdge(int i, int j);
    PixelNode* bottomEdge(int i, int j);
    PixelNode* topEdge(int i, int j);

    std::pair<int,int> getNearestNeighbour(int i, int j);

};


#endif //IMAGEMATTING_PIXELGRAPH_H
