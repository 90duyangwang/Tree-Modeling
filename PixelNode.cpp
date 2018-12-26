//
// Created by premsai chinthamreddy on 03/11/18.
//

#include "PixelNode.h"
PixelNode::PixelNode(int x, int y):x(x),y(y){}

void PixelNode::setCoordinates(int x, int y) { this->x = x; this->y = y;}

void PixelNode::setAttracPoint(bool val) { this->isAttracPoint = val; }

void PixelNode::setDirecPoint(bool val) { this->isDirecPoint = val; }

void PixelNode::setNearestNode(PixelNode *node) { this->nearestNode = node; }
