#include "morphologicaloperationclass.h"
#include <opencv2/opencv.hpp>
using namespace cv;

void morphologicalOperationClass::run()
{
    Mat element = getStructuringElement( MORPH_RECT, Size( kernelSize,kernelSize));
    morphologyEx(img, imgResult, MORPH_OPEN, element );
    morphologyEx( imgResult, imgResult, MORPH_CLOSE, element );
    imgResultColor = Mat(imgReference.rows,imgReference.cols,CV_8UC3);
    for(int y=0;y<imgReference.rows;y++)
        for(int x=0;x<imgReference.cols;x++)
        {
            if(imgResult.at<float>(y,x)==255) // jika label nukleus
                imgResultColor.at<Vec3b>(y,x)= imgReference.at<Vec3b>(y,x); //diberi warna sesuai original
            else
                imgResultColor.at<Vec3b>(y,x) = {255,255,255}; //jika tidak diberi warna
        }
}

