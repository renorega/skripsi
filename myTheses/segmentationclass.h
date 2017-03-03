#ifndef SEGMENTATIONCLASS_H
#define SEGMENTATIONCLASS_H
#include <opencv2/opencv.hpp>
using namespace cv;

class segmentationClass
{
private:
    Mat img,imgResult,imgResultColor,bestLabels,centers;
    vector<Mat> imgK;
    vector<float> sat;
    float highestSatValue;
    int K, highestSatIndex;
public:
    segmentationClass(Mat img,int K=4)
    {
        this->img = img;
        this->K = K;
    }
    void run();
    void findHighestSaturation();
    void assignColor();
    Mat getResult(){return imgResult;}
    Mat getResultColor(){return imgResultColor;}
};

#endif // SEGMENTATIONCLASS_H
