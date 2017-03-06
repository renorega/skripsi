#include "watershedclass.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
using namespace std;
using namespace cv;

void watershedClass::run()
{
    imgColor = Mat(img.rows,img.cols,CV_8UC3);
    for(int y=0;y<imgReference.rows;y++){
        for(int x=0;x<imgReference.cols;x++){
            if(img.at<float>(y,x)==255) // jika label nukleus
                imgColor.at<Vec3b>(y,x)= imgReference.at<Vec3b>(y,x); //diberi warna sesuai original

           else
                imgColor.at<Vec3b>(y,x) = {255,255,255}; //jika tidak diberi warna
        }
    }

    Mat imgDistInput, imgDistTransform, dist_8u;

    img.convertTo(imgDistInput,CV_8UC3);
    distanceTransform(imgDistInput, imgDistTransform, CV_DIST_L2, 5);
    normalize(imgDistTransform, imgDistTransform, 0, 1., NORM_MINMAX);
    threshold(imgDistTransform, imgDistTransform, thresholdValueBase, 1., CV_THRESH_BINARY);
    imgDistTransform.convertTo(dist_8u, CV_8U);
    findContours(dist_8u, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
//    cout << "contours size: " << contours[0][0].x<< endl;
    markers = Mat::zeros(imgDistTransform.size(), CV_32SC1); //SC1 -> Signed Char 1channel
    for (size_t i = 0; i < contours.size(); i++)
        drawContours(markers, contours, static_cast<int>(i), Scalar::all(static_cast<int>(i)+1), -1);

    circle(markers, Point(5,5), 3, CV_RGB(255,255,255), -1);
    watershed(imgColor, markers);

    imgResultColor = Mat(img.rows,img.cols,CV_8UC3);
    imgResult= Mat::zeros(img.size(), CV_8UC1);
    for (int i = 0; i < imgReference.rows; i++)
    {
        for (int j = 0; j < imgReference.cols; j++)
        {
            int index = markers.at<int>(i,j);
            if (index > 0 && index <= static_cast<int>(contours.size()))
                imgResult.at<char>(i,j) = 255;
            else
                imgResult.at<char>(i,j) = 0;
        }
    }

    for (int i = 0; i < imgReference.rows; i++)
    {
        for (int j = 0; j < imgReference.cols; j++)
        {
            if (imgResult.at<char>(i,j))
                imgResultColor.at<Vec3b>(i,j) = imgColor.at<Vec3b>(i,j) ;
            else
                imgResultColor.at<Vec3b>(i,j) =Vec3b(255,255,255);
        }
    }
}
void watershedClass::calculateArea()
{
    area.resize(contours.size()+1);
    for (int row=0;row<markers.rows;row++)
        for(int col=0;col<markers.cols;col++)
            // because background are not included in contours , so we can use <contours.size()+1 or <=contours.size()
            for(int index=0;index<=static_cast<int>(contours.size());index++)
                if(markers.at<int>(row,col)==index)
                    area[index]++;
    cout << "Print area from first WT!" << endl;
    for(int i=0; i<=static_cast<int>(contours.size());i++)
        cout << "Area-" << i << ": " << area[i] << endl;
}
void watershedClass::findConnectedNuclei()
{
    for(int i=0; i<=static_cast<int>(contours.size());i++)
    // If background are extracted into WT region, so region with more than 100000 pixels are ignored
        if(area[i]>thresholdNuclei && area[i]<100000)
            indexConnectedNuclei.push_back(i);

    if(indexConnectedNuclei.size()>0)
        isSecondWT = true;

    if(isSecondWT)
    {
        for(int i=0;i<static_cast<int>(indexConnectedNuclei.size());i++)
            connectedNuclei = connectedNuclei + (markers==indexConnectedNuclei[i]);

        connectedNucleiColor = Mat(imgColor.rows,imgColor.cols,CV_8UC3);
        for(int y=0;y<imgReference.rows;y++)
            for(int x=0;x<imgReference.cols;x++)
            {
                if(connectedNuclei.at<unsigned char>(y,x)) // jika ada warna atau tidak sama dengan 0
                    connectedNucleiColor.at<Vec3b>(y,x)= imgColor.at<Vec3b>(y,x); //diberi warna sesuai original

                else
                    connectedNucleiColor.at<Vec3b>(y,x) = Vec3b{255,255,255}; //jika tidak diberi warna
            }

        cout << "\nPerforming second WT . . ." << endl;
        for (int i:indexConnectedNuclei)
        {
            int tempTarget = round(area[i]/areaNucleiIndividu);
            targetRegionConnectedNuclei.push_back(tempTarget);
        }
        // Print indexConnectedNuclei and its area
        for (int i=0;i<static_cast<int>(indexConnectedNuclei.size());i++)
            cout << "Index connected nuclei: " << indexConnectedNuclei[i] << " Area connected nuclei : " << area[indexConnectedNuclei[i]] << " Target region: " << targetRegionConnectedNuclei[i] << endl;
        runSecondWT();
    }
}
void watershedClass::runSecondWT()
{
    extractConnectedNuclei();
    findValue();
    performSecondWatershed();
}
void watershedClass::extractConnectedNuclei()
{
    for(int i=0;i<static_cast<int>(indexConnectedNuclei.size());i++)
        connectedNuclei = connectedNuclei + (markers==indexConnectedNuclei[i]);

//    showImage("Connected Nuclei",connectedNuclei);

    connectedNucleiColor = Mat(imgColor.rows,imgColor.cols,CV_8UC3);
    for(int y=0;y<imgReference.rows;y++)
        for(int x=0;x<imgReference.cols;x++)
        {
            if(connectedNuclei.at<unsigned char>(y,x)) // jika ada warna atau tidak sama dengan 0
                connectedNucleiColor.at<Vec3b>(y,x)= imgColor.at<Vec3b>(y,x); //diberi warna sesuai original

            else
                connectedNucleiColor.at<Vec3b>(y,x) = Vec3b{255,255,255}; //jika tidak diberi warna
        }
}
void watershedClass::findValue()
{
    for(int i=0;i<static_cast<int>(indexConnectedNuclei.size());i++)
    {
        cout <<"Performing the " << i << " index"<<endl;
        int targetContour = targetRegionConnectedNuclei[i];
        cout <<"Target contour of-" <<i<<":" << targetContour << endl;

        Mat connectedNuclei= markers== indexConnectedNuclei[i];
        Mat connectedNucleiDistInput;
        connectedNuclei.convertTo(connectedNucleiDistInput,CV_8U);
        Mat connectedNucleiDistTransform;
        distanceTransform(connectedNucleiDistInput, connectedNucleiDistTransform, CV_DIST_L2, 5);
        normalize(connectedNucleiDistTransform, connectedNucleiDistTransform, 0, 1., NORM_MINMAX);

        thresholdValue = thresholdValueBase+0.05; // This value for thresholding second WT
        Mat connectedNucleiPeaks;
        threshold(connectedNucleiDistTransform, connectedNucleiPeaks, thresholdValue, 1., CV_THRESH_BINARY);

        Mat connectedNuclei8u;
        connectedNucleiPeaks.convertTo(connectedNuclei8u, CV_8U);

        // Find total markers
        vector<vector<Point>> contoursConnectedNuclei;
        findContours(connectedNuclei8u, contoursConnectedNuclei, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

        // Parameter for finding threshold value for distant transform
        bool successWT= false, isLargerSize= false;
        int largerSize = 0;
        float thresholdValueBig = 0;

        vector<vector<Point>> contoursConnectedNucleiLarger;
        Mat connectedNucleiPeaksLarger;

        if(static_cast<int>(contoursConnectedNuclei.size())==targetContour)
        {
            listContoursConnectedNuclei.insert(listContoursConnectedNuclei.end(),contoursConnectedNuclei.begin(),contoursConnectedNuclei.end());
            successWT = true;
        }

        while(static_cast<int>(contoursConnectedNuclei.size())!=targetContour)
        {
            if(thresholdValue>=1.0) break;
            /* if (thresholdValue>=1/0
            {
                targetContour--;
                doWhile
            }
            */

//            // If contours size is larger than targetContour but smaller than largerSize
//            if(static_cast<int>(indexConnectedNuclei.size())>targetContour && static_cast<int>(indexConnectedNuclei.size())<largerSize && !isLargerSize)
//            {
//                //contoursConnectedNucleiLarger = contoursConnectedNuclei;
//                //connectedNucleiPeaksLarger = connectedNucleiPeaks;
//                isLargerSize = true;
//                largerSize = contoursConnectedNuclei.size();
//                thresholdValueBig = thresholdValue;
//            }

            thresholdValue = thresholdValue + 0.05;
            threshold(connectedNucleiDistTransform, connectedNucleiPeaks, thresholdValue, 1., CV_THRESH_BINARY);

            connectedNucleiPeaks.convertTo(connectedNuclei8u, CV_8U);

            contoursConnectedNuclei.clear(); // clear vector
            findContours(connectedNuclei8u, contoursConnectedNuclei, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

            if(static_cast<int>(contoursConnectedNuclei.size())==targetContour)
            {
                listContoursConnectedNuclei.insert(listContoursConnectedNuclei.end(),contoursConnectedNuclei.begin(),contoursConnectedNuclei.end());
                successWT = true;
                break;
            }
        }

        /*
        if(!successWT && isLargerSize )
        {
            thresholdValue = thresholdValueBig;
            threshold(connectedNucleiDistTransform, connectedNucleiPeaks, thresholdValue, 1., CV_THRESH_BINARY);
            connectedNucleiPeaksLarger.convertTo(connectedNuclei8u, CV_8U);
            contoursConnectedNuclei = contoursConnectedNucleiLarger;
            //contoursConnectedNuclei.clear(); // clear vector
            //cout << contoursConnectedNuclei.size() << endl;
            //findContours(connectedNuclei8u, contoursConnectedNuclei, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
        }
        */

        //if(!successWT && !isLargerSize)
        if(!successWT)
            cout << "This is single nucleus!" << endl;

        // END OF NEW SECOND WT
        cout <<"Final threshold value for index-" << indexConnectedNuclei[i] << " is: " <<thresholdValue << endl;
        allConnectedNucleiPeaks = allConnectedNucleiPeaks + (connectedNucleiPeaks!=0);
    }
}
void watershedClass::performSecondWatershed()
{
    markersConnectedNuclei = Mat::zeros(allConnectedNucleiPeaks.size(), CV_32SC1); //SC1 -> Signed Char 1channel

    // Draw the foreground markers
      for (size_t i = 0; i < listContoursConnectedNuclei.size(); i++)
            drawContours(markersConnectedNuclei, listContoursConnectedNuclei, static_cast<int>(i), Scalar::all(static_cast<int>(i)+1), -1);
    // Draw the background marker
    circle(markersConnectedNuclei, Point(5,5), 3, CV_RGB(255,255,255), -1);
    watershed(connectedNucleiColor,markersConnectedNuclei);

    for (int i = 0; i < imgReference.rows; i++)
        for (int j = 0; j < imgReference.cols; j++)
        {
            int index = markersConnectedNuclei.at<int>(i,j);
            if(index<0)
            {
                imgResultColor.at<Vec3b>(i,j) =Vec3b(255,255,255);
                imgResult.at<char>(i,j) = 0;
            }
        }
}
