#ifndef MORPHOLOGICALOPERATIONCLASS_H
#define MORPHOLOGICALOPERATIONCLASS_H
#include <opencv2/opencv.hpp>
using namespace cv;

class morphologicalOperationClass
{
private:
    Mat img, imgReference, imgResult,imgResultColor;
    int kernelSize;
public:
    morphologicalOperationClass(Mat img,Mat imgReference,int kernelSize=7)
    {
        this->img = img;
        this->kernelSize = kernelSize;
        this->imgReference = imgReference;
    }
    void run();
    Mat getResult(){return imgResult;}
    Mat getResultColor(){return imgResultColor;}
};

#endif // MORPHOLOGICALOPERATIONCLASS_H
