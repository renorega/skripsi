// ---------------------------- HEADER FILE -----------------------------------
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include "morphologicaloperationclass.h"
#include "segmentationclass.h"
#include "watershedclass.h"
using namespace std;
using namespace cv;

// ----------------------------- FUNCTION DECLARATION------------------------------
Mat preprocessing(Mat img);
void showImage(string name,Mat img);

// ----------------------------------- MAIN FUNCTION--------------------------------------
int main()
{
    //1. READ IMAGE
    string location = "/home/reno/skripsi/ALL_SAMPLES/ALL_Sardjito/gambar_29mei/AfarelAzis_17april_01680124/";
    //string location = "/home/reno/skripsi/ALL_SAMPLES/ALL_Sardjito/gambar_29mei/sugesti_01684902_26mei/";
    string nameFile= "1-4.jpg";
    cout <<"File: " << nameFile << endl;
    Mat imgOriginal = imread(location+nameFile,IMREAD_COLOR);
    showImage("Original",imgOriginal);
    cout << "img rows : " <<  imgOriginal.rows << " img cols :" << imgOriginal.cols << endl;

    //2. PERFORM PREPROCESSING
    Mat imgPreprocessing= preprocessing(imgOriginal);

    //3. PERFORM SEGMENTATION
    segmentationClass segment(imgPreprocessing,imgOriginal);
    segment.run();
    Mat imgKMeans = segment.getResult();
    showImage("Kmeans Binary",imgKMeans);
    showImage("KMeans Color",segment.getResultColor());

    //4. PERFORM MORPHOLOGICAL OPENING AND CLOSING
    morphologicalOperationClass morph(imgKMeans,imgOriginal);
    morph.run();
    showImage("Img morph",morph.getResultColor());
    //5. PERFORMING WATERSHED TRANSFORMATION
    watershedClass wt(morph.getResult(),imgOriginal,0.3);
    wt.run();
    showImage("First WT Result",wt.getResultColor());
    wt.calculateArea();

    wt.setThresholdNuclei(28000);
    wt.setAreaNucleiIndividu(20000);
    wt.findConnectedNuclei();
    if(wt.getIsSecondWT())
    {
        showImage("Connected Nuclei",wt.getConnectedNuclei());
        showImage("Final WT Result",wt.getResultColor());
    }
    waitKey();
    destroyAllWindows();
    return 0;
}

//-------------------------------FUNCTIONS DEFINITION-----------------------------------

void showImage(string name,Mat img)
{
    namedWindow(name,WINDOW_NORMAL);
    imshow(name,img);
}

Mat preprocessing(Mat img)
{
    // Convert input img into HSV
    Mat imgHSV;
    cvtColor(img,imgHSV,CV_BGR2HSV);

    // Perform histogram equalization in channel V
    vector<Mat> vectHSV;
    split(imgHSV, vectHSV);
    Mat imgHist;
    equalizeHist(vectHSV[2],vectHSV[2]);
    Mat imgEqualized;
    merge(vectHSV,imgEqualized);
    return imgEqualized;
}
