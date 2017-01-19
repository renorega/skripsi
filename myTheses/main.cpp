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
    /*
    namedWindow("BGR2HSV",WINDOW_NORMAL);
    imshow("BGR2HSV",imgHSV);
    namedWindow("BGR2HSVFULL",WINDOW_NORMAL);
    imshow("BGR2HSVFULL",imgHSVFull);
    */

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

    /*
    //Show the result of Histogram Equalization
    namedWindow("Equalized Image",WINDOW_NORMAL);
    imshow("Equalized Image",imgEqualized);
    namedWindow("Equalized Image Full",WINDOW_NORMAL);
    imshow("Equalized Image Full",imgEqualizedFull);
    */

    // Perform median filtering
    // Trying median filter 7x7
    Mat imgMedian,imgMedianFull;
    medianBlur(imgEqualized,imgMedian,7);
    medianBlur(imgEqualizedFull,imgMedianFull,7);

    // show result of median filtering 7x7
    /*
    namedWindow("Image Median",WINDOW_NORMAL);
    imshow("Image Median",imgMedian);
    namedWindow("Image Median Full",WINDOW_NORMAL);
    imshow("Image Median Full",imgMedianFull);
    */

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

    //buat vector bgr tipe Mat
    vector<Mat> imgK,imgKFull;

    split(imgMedian, imgK);
    split(imgMedianFull, imgKFull);

    // show S-channel from images
    /*
    namedWindow("S",WINDOW_NORMAL);
    imshow("S", imgK[1]);
    namedWindow("SFull",WINDOW_NORMAL);
    imshow("SFull", imgKFull[1]);
    */

    //Aplikasikan algoritma KMEANS
    //buat matrix P kosong dengan ukuran row = column x row, col=1 dan tipe CV_32F
    Mat p = Mat::zeros(img.cols*img.rows, 1, CV_32F);
    Mat p2 = Mat::zeros(img.cols*img.rows, 1, CV_32F);
    Mat bestLabels,bestLabels2,centers,centers2, clustered,clustered2;

    for(int i=0; i<img.cols*img.rows; i++) {
        p.at<float>(i,0) = imgK[1].data[i] / 255.0; // how to use only one channel (S)
    }

    for(int i=0; i<img.cols*img.rows; i++) {
        p2.at<float>(i,0) = imgKFull[1].data[i] / 255.0; // how to use only one channel (S)
    }

    int K = 4;
    kmeans(p, K, bestLabels,
            TermCriteria( CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 10, 1.0),
            3, KMEANS_PP_CENTERS, centers);

    kmeans(p2, K, bestLabels2,
            TermCriteria( CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 10, 1.0),
            3, KMEANS_PP_CENTERS, centers2);

    //buat ngasih warna
    int colors[K]; //0 adalah hitam, 255 adalah putih

    // Untuk mencari gambar dengan rata-rata S tertinggi
    /*
    for(int i=0;i<K;i++)
    {
        colors[i] = 0;
        int x=0;
        while(x!=i && x<K)
        {
            colors[x]=255;
            x++;
        }

        clustered = Mat(img.rows, img.cols, CV_32F);
        for(int i=0; i<img.cols*img.rows; i++) {
            clustered.at<float>(i/img.cols, i%img.cols) = (float)(colors[bestLabels.at<int>(0,i)]);
        }
        for(int y=0;y<img.rows;y++){
            for(int x=0;x<img.cols;x++){
                if(clustered.at<float>(y,x)==0)
                {
                    imgKFinal.at<Vec3b>(y,x)= imgK.at<Vec3b>(y,x);
                }

               else
                {
                    imgKFinal.at<Vec3b>(y,x) = {255,255,255};
                }
            }
        }
    }
    */
    colors[0] = 255;
    colors[1] = 0;
    colors[2] = 255;
    colors[3] = 255;

    // Ngereshape plus ngasih warna dengan best labels
    clustered = Mat(img.rows, img.cols, CV_32F);
    for(int i=0; i<img.cols*img.rows; i++) {
        clustered.at<float>(i/img.cols, i%img.cols) = (float)(colors[bestLabels.at<int>(0,i)]);
    }

    clustered2 = Mat(img.rows, img.cols, CV_32F);
    for(int i=0; i<img.cols*img.rows; i++) {
        clustered2.at<float>(i/img.cols, i%img.cols) = (float)(colors[bestLabels2.at<int>(0,i)]);
    }

    // Mengembalikan warna dari img original yang memiliki 3 channel, convert image ke 3-ch img
    Mat imgKFinal(img.rows,img.cols,CV_8UC3);
    Mat imgKFinalFull(img.rows,img.cols,CV_8UC3);
    for(int y=0;y<img.rows;y++){
        for(int x=0;x<img.cols;x++){
            if(clustered.at<float>(y,x)==0)
            {
                imgKFinal.at<Vec3b>(y,x)= imgK[1].at<Vec3b>(y,x);
            }

           else
            {
                imgKFinal.at<Vec3b>(y,x) = {255,255,255};
            }
        }
    }

    for(int y=0;y<img.rows;y++){
        for(int x=0;x<img.cols;x++){
            if(clustered2.at<float>(y,x)==0)
            {
                imgKFinalFull.at<Vec3b>(y,x)= imgKFull[1].at<Vec3b>(y,x);
            }

           else
            {
                imgKFinalFull.at<Vec3b>(y,x) = {255,255,255};
            }
        }
    }

    // show the result of img with binary color
    /*
    namedWindow("clustered",WINDOW_NORMAL);
    imshow("clustered", clustered);
    namedWindow("clusteredFull",WINDOW_NORMAL);
    imshow("clusteredFull", clustered2);
    */

    // Show the result of img when assigned with original color
    namedWindow("clustered",WINDOW_NORMAL);
    imshow("clustered", imgKFinal);
    namedWindow("clusteredFull",WINDOW_NORMAL);
    imshow("clusteredFull", imgKFinalFull);

    waitKey();
    destroyAllWindows();
    return 0;
}
