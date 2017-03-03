#ifndef WATERSHEDCLASS_H
#define WATERSHEDCLASS_H
#include <opencv2/opencv.hpp>
using namespace cv;

class watershedClass
{
private:
    Mat img,imgReference,imgColor,imgResult,imgResultColor,markers;
    const float thresholdValueBase;
    float thresholdValue;
    vector<vector<Point>> contours;
    vector<float> area,targetRegionConnectedNuclei;
    vector<int> indexConnectedNuclei;
    bool isSecondWT = false;
    int thresholdNuclei;
    float areaNucleiIndividu;

    // from second WT
    Mat allConnectedNucleiPeaks;
    vector<vector<Point>>listContoursConnectedNuclei;
    Mat markersConnectedNuclei;
    Mat connectedNuclei,connectedNucleiColor;

public:
    watershedClass(Mat img,Mat imgReference,const float _thresholdValueBase=0.4) : thresholdValueBase(_thresholdValueBase)
    {
        this->img = img;
        this->imgReference = imgReference;
        //this->thresholdValue = thresholdValue;
    }
    void run();
    void calculateArea();
    void findConnectedNuclei();
    void runSecondWT();
    void extractConnectedNuclei();
    void findValue();
    void performSecondWatershed();

    void setThresholdNuclei(int threshold){thresholdNuclei = threshold;}
    void setAreaNucleiIndividu(float areaNuclei){areaNucleiIndividu = areaNuclei;}

    bool getIsSecondWT(){return isSecondWT;}
    Mat getResult(){return imgResult;}
    Mat getResultColor(){return imgResultColor;}
    Mat getConnectedNuclei(){return connectedNuclei;}
    Mat getConnectedNucleiColor(){return connectedNucleiColor;}
};

#endif // WATERSHEDCLASS_H
