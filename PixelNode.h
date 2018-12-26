//
// Created by premsai chinthamreddy on 03/11/18.
//

#ifndef IMAGEMATTING_PIXELNODE_H
#define IMAGEMATTING_PIXELNODE_H

typedef unsigned char uchar;

class PixelNode {
public:
    int x; // X coordinate of the pixel in the image
    int y; // Y coordinate of the pixel in the image
    bool isAttracPoint = false;
    bool isDirecPoint = false;
    bool isVisited = false;
    PixelNode* nearestNode;

    PixelNode(int x, int y);
    void setCoordinates(int x, int y);
    void setAttracPoint(bool val);
    void setDirecPoint(bool val);
    void setNearestNode(PixelNode* node);
};


#endif //IMAGEMATTING_PIXELNODE_H
