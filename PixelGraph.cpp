//
// Created by premsai chinthamreddy on 03/11/18.
//

#include "PixelGraph.h"
#include <queue>

PixelGraph::PixelGraph(int h, int w): height(h), width(w){
    this->node_matrix.resize(h, std::vector<PixelNode*>(w));
};
PixelNode* PixelGraph::get(int i, int j){
    if(i<0 && i>height && j<0 && j>width){
        return NULL;
    }
    return node_matrix[i][j];
}
void PixelGraph::set(int i, int j, PixelNode* p) {
    if(!(i<0 && i>width && j<0 && j>width)){
        node_matrix[i][j] = p;
    }
}
PixelNode* PixelGraph::leftEdge(int i, int j) {
    if(j<1 && j>width){
        return NULL;
    }else {
        return node_matrix[i][j-1];
    }
}
PixelNode* PixelGraph::rightEdge(int i, int j) {
    if(j<0 && j>= width-1) {
        return NULL;
    }else{
        return node_matrix[i][j+1];
    }
}
PixelNode* PixelGraph::topEdge(int i, int j) {
    if(i<1 && i>height){
        return NULL;
    }else {
        return node_matrix[i-1][j];
    }

}
PixelNode* PixelGraph::bottomEdge(int i, int j) {
    if(i<0 && i>=height-1){
        return NULL;
    }else {
        return node_matrix[i+1][j];
    }

}

std::pair<int,int> PixelGraph::getNearestNeighbour(int i, int j) {
    std::pair<int,int> p(i,j);
    std::queue<std::pair<int,int>> q;
    q.push(p);
    while(!q.empty()){
        std::pair<int,int> n = q.front();
        q.pop();
        if(this->get(n.first, n.second) != NULL){
            this->get(n.first, n.second)->isVisited = true;
            if(this->get(n.first, n.second)->isAttracPoint) {
                return std::make_pair(n.first, n.second);
            }else {
                int l = n.first;
                int m = n.second;

                if(this->get(l-1, m) && !this->get(l-1, m)->isVisited){
                    q.push(std::make_pair(l-1, m));
                }
                if(this->get(l+1, m) && !this->get(l+1, m)->isVisited){
                    q.push(std::make_pair(l+1, m));
                }
                if(this->get(l, m-1) && !this->get(l, m-1)->isVisited){
                    q.push(std::make_pair(l, m-1));
                }
                if(this->get(l, m+1) && !this->get(l, m+1)->isVisited){
                    q.push(std::make_pair(l, m+1));
                }
            }
        }
    }
    return std::make_pair(-1,-1);
}