#include "segmentationclass.h"
#include <opencv2/opencv.hpp>
using namespace cv;

void segmentationClass::run()
{
    split(img,imgK);
    Mat p = Mat::zeros(img.cols*img.rows, 1, CV_32F);
    for(int i=0; i<img.cols*img.rows; i++)
        p.at<float>(i,0) = imgK[1].data[i] / 255.0; // Use only one channel(S) with normalization to ease computation
    kmeans(p, K, bestLabels,
        TermCriteria( CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 10, 1.0),
        3, KMEANS_PP_CENTERS, centers);
    findHighestSaturation();
    assignColor();
}
void segmentationClass::findHighestSaturation()
{
    sat.resize(K);
    vector<int> div(K);
    for (int label=0;label<K;label++)
    {
        for(int i=0;i<img.rows*img.cols;i++)
        {
            if(bestLabels.at<int>(i,0)==label)
            {
                sat[label] = sat[label] + imgK[1].data[i]/255.0; // calculate total saturation
                div[label] +=1; // calculate area of label
            }
        }
        sat[label] = sat[label]/div[label]; // calculate mean value of saturation
    }

    // Find labels with highest saturation
    highestSatValue = sat[0]; // store highest saturation value
    highestSatIndex= 0; // store labels with highest saturation
    for(int i=1;i<K;i++)
        if(highestSatValue<sat[i])
        {
            highestSatValue= sat[i];
            highestSatIndex= i;
        }
}
void segmentationClass::assignColor()
{
    // Assign labels with white
    int colors[K]; //0 is black, 255 is white
    for(int i=0;i<K;i++)
    {
        if(i==highestSatIndex) colors[i]=255;
        else colors[i]=0;
    }

    // Reshape and give color
    imgResult= Mat(img.rows, img.cols, CV_32F);
    for(int i=0; i<img.cols*img.rows; i++)
        imgResult.at<float>(i/img.cols, i%img.cols) = (float)(colors[bestLabels.at<int>(0,i)]);
}
