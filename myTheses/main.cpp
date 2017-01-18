#include <opencv2/opencv.hpp>
#include <iostream>
using namespace std;
using namespace cv;

int main(int argc, char *argv[])
{
    //loading image
    Mat img,imgHSV,imgHSVFull;
    // use IMREAD_COLOR to access image in BGR format as 8 bit image
    img = imread("/home/reno/skripsi/ALL_SAMPLES/ALL_Sardjito/gambar_29mei/AfarelAzis_17april_01680124/1-4.jpg",IMREAD_COLOR);
    namedWindow("Original",WINDOW_NORMAL);
    imshow("Original",img);

    // convert image into HSV
    cvtColor(img,imgHSV,CV_BGR2HSV);
    cvtColor(img,imgHSVFull,CV_BGR2HSV_FULL);

    // show HSV images
    namedWindow("BGR2HSV",WINDOW_NORMAL);
    imshow("BGR2HSV",imgHSV);
    namedWindow("BGR2HSVFULL",WINDOW_NORMAL);
    imshow("BGR2HSVFULL",imgHSVFull);

    // split images into 3 channel to obtain only V-Channel for hist. equalization
    vector<Mat> vectHSV, vectHSVFull;
    split(imgHSV, vectHSV);
    split(imgHSVFull, vectHSVFull);

    // These lines is for showing channel V from imgHSV
    /*
    namedWindow("V-Channel HSV",WINDOW_NORMAL);
    imshow("V-Channel HSV",vectHSV[2]);
    namedWindow("V-Channel HSVFull",WINDOW_NORMAL);
    imshow("V-Channel HSVFull",vectHSVFull[2]);
    */

    // perform histogram equalization in channel-V
    Mat imgHist, ImgHistFull;
    equalizeHist(vectHSV[2],vectHSV[2]);
    equalizeHist(vectHSVFull[2],vectHSVFull[2]);

    // This line for showing the result of histogram equalization in channel V
    /*
    namedWindow("Equalized V-Channel HSV",WINDOW_NORMAL);
    imshow("Equalized V-Channel HSV",vectHSV[2]);
    namedWindow("Equalized V-Channel HSVFull",WINDOW_NORMAL);
    imshow("Equalized V-Channel HSVFull",vectHSVFull[2]);
    */

    // merge equalized V-Channel to Image
    Mat imgEqualized, imgEqualizedFull;
    merge(vectHSV,imgEqualized);
    merge(vectHSVFull,imgEqualizedFull);

    //Show the result of Histogram Equalization
    namedWindow("Equalized Image",WINDOW_NORMAL);
    imshow("Equalized Image",imgEqualized);
    namedWindow("Equalized Image Full",WINDOW_NORMAL);
    imshow("Equalized Image Full",imgEqualizedFull);

    // Perform median filtering
    // Trying median filter 7x7
    Mat imgMedian,imgMedianFull;
    medianBlur(imgEqualized,imgMedian,7);
    medianBlur(imgEqualizedFull,imgMedianFull,7);
    namedWindow("Image Median",WINDOW_NORMAL);
    imshow("Image Median",imgMedian);
    namedWindow("Image Median Full",WINDOW_NORMAL);
    imshow("Image Median Full",imgMedianFull);

    // Trying median filter 3x3
    /*
    Mat imgMedian2,imgMedianFull2;
    medianBlur(imgEqualized,imgMedian2,3);
    medianBlur(imgEqualizedFull,imgMedianFull2,3);
    namedWindow("Image Median2",WINDOW_NORMAL);
    imshow("Image Median2",imgMedian2);
    namedWindow("Image Median Full2",WINDOW_NORMAL);
    imshow("Image Median Full2",imgMedianFull2);
    */
    waitKey();
    destroyAllWindows();
    return 0;
}
