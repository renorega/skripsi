#include <opencv2/opencv.hpp>
#include <iostream>
using namespace std;
using namespace cv;

int main(int argc, char *argv[])
{
    //loading image
    Mat img,imgHSV;

    // use IMREAD_COLOR to access image in BGR format as 8 bit image
    img = imread("/home/reno/skripsi/ALL_SAMPLES/ALL_Sardjito/gambar_29mei/AfarelAzis_17april_01680124/24-28.jpg",IMREAD_COLOR);
    namedWindow("Original",WINDOW_NORMAL);
    imshow("Original",img);

    // convert image into HSV
    // HSV_FULL gives Hue value between 0-255, while HSV gives Hue value between 0-180 (H/2)
    // (A 8-bits img (0-255) can't contain hue value 0-360)
    cvtColor(img,imgHSV,CV_BGR2HSV);

    /*
    // show HSV images
    namedWindow("BGR2HSV",WINDOW_NORMAL);
    imshow("BGR2HSV",imgHSV);
    */

    // split images into 3 channel to obtain only V-Channel for hist. equalization
    vector<Mat> vectHSV;
    split(imgHSV, vectHSV);

    /*
    // These lines is for showing channel V from imgHSV
    namedWindow("V-Channel HSV",WINDOW_NORMAL);
    imshow("V-Channel HSV",vectHSV[2]);
    */

    // perform histogram equalization in channel-V
    Mat imgHist;
    equalizeHist(vectHSV[2],vectHSV[2]);

    /*
    // This line for showing the result of histogram equalization in channel V
    namedWindow("Equalized V-Channel HSV",WINDOW_NORMAL);
    imshow("Equalized V-Channel HSV",vectHSV[2]);
    */

    // merge equalized V-channel to Image
    Mat imgEqualized;
    merge(vectHSV,imgEqualized);

    /*
    //Show the result of Histogram Equalization
    namedWindow("Equalized Image",WINDOW_NORMAL);
    imshow("Equalized Image",imgEqualized);
    */

    // Perform median filtering
    // Trying median filter 7x7
    /*
    Mat imgMedian;
    medianBlur(imgEqualized,imgMedian,7);
    // show result of median filtering 7x7
    /*
    namedWindow("Image Median",WINDOW_NORMAL);
    imshow("Image Median",imgMedian);
    */

    // Trying median filter 3x3
    /*
    Mat imgMedian2;
    medianBlur(imgEqualized,imgMedian2,3);
    namedWindow("Image Median2",WINDOW_NORMAL);
    imshow("Image Median2",imgMedian2);
    */

    //buat vector bgr tipe Mat
    vector<Mat> imgK;
    // split every channel of image filtered with median
    split(imgEqualized, imgK);

    /*
    // show S-channel from images
    namedWindow("S",WINDOW_NORMAL);
    imshow("S", imgK[1]);
    */

    //Aplikasikan algoritma KMEANS
    //buat matrix P kosong dengan ukuran row = column x row, col=1 dan tipe CV_32F
    Mat p = Mat::zeros(img.cols*img.rows, 1, CV_32F);
    Mat bestLabels,centers,imgKMeansGrayscale;
    for(int i=0; i<img.cols*img.rows; i++) {
        p.at<float>(i,0) = imgK[1].data[i] / 255.0; // Use only one channel(S) with normalization to ease computation
    }

    // perform Kmeans clustering
    int K = 4;
    kmeans(p, K, bestLabels,
            TermCriteria( CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 10, 1.0),
            3, KMEANS_PP_CENTERS, centers);

    // Mencari label dengan rata-rata nilai S tertinggi
    float sat[K] = {0,0,0,0}; // variable untuk mencari nilai S tertinggi
    int div[K] = {0,0,0,0}; // untuk menghitung total area dari label
    for (int label=0;label<K;label++)
    {
        for(int i=0;i<img.rows*img.cols;i++)
        {
            if(bestLabels.at<int>(i,0)==label) // jika label i maka
            {
                sat[label] = sat[label] + imgK[1].data[i]/255.0; // hitung total nilai S-channel
                div[label] +=1; // hitung area dari label
            }
        }
        sat[label] = sat[label]/div[label]; // hitung rata-rata nilai S
    }

    // mencari label nukleus melalui nilai S tertinggi
    float maxSat = sat[0]; // variabel untuk menyimpan nilai saturasi tertinggi
    int indexSat = 0; // variabel untuk menyimpan label nukleus yaitu label yang memiliki nilai saturasi tertinggi
    for(int i=1;i<K;i++)
    {
        if(maxSat<sat[i]) // jika lebih besar maka itulah nilai yang lebih tinggi
        {
            maxSat = sat[i];
            indexSat = i; // menyimpan label nukleus
        }
    }

    //buat ngasih warna 255 atau putih ke label nukleus
    int colors[K]; //0 adalah hitam, 255 adalah putih
    for(int i=0;i<K;i++)
    {
        if(i==indexSat) colors[i]=255; // jika label nukleus diberi warna putih
        else colors[i]=0;
    }

    // Ngereshape plus ngasih warna dengan best labels
    imgKMeansGrayscale= Mat(img.rows, img.cols, CV_32F);
    for(int i=0; i<img.cols*img.rows; i++) {
        imgKMeansGrayscale.at<float>(i/img.cols, i%img.cols) = (float)(colors[bestLabels.at<int>(0,i)]);
    }

    // Show the result of KMeans with grayscale
    namedWindow("Grayscale KMeans",WINDOW_NORMAL);
    imshow("Grayscale KMeans", imgKMeansGrayscale);

    // Perform Morphological Opening
    Mat imgMorphOpen;
    int kernelSize = 7;
    Mat element = getStructuringElement( MORPH_RECT, Size( kernelSize,kernelSize));
    morphologyEx(imgKMeansGrayscale, imgMorphOpen, MORPH_OPEN, element );
    namedWindow("ImgMorphOpen",CV_WINDOW_NORMAL);
    imshow( "ImgMorphOpen", imgMorphOpen);

    // Perform Morphological Closing
    /*
    kernelSize = 3;
    Mat imgMorphClose, imgMorphFinal;
    element = getStructuringElement( MORPH_RECT, Size( kernelSize,kernelSize));
    morphologyEx( clustered, imgMorphClose, MORPH_CLOSE, element );
    namedWindow("ImgMorphClose",CV_WINDOW_NORMAL);
    imshow( "ImgMorphClose", imgMorphClose);

    // Perform Morphological Opening + Closing
    morphologyEx( imgMorphOpen, imgMorphFinal, MORPH_CLOSE, element );
    namedWindow("ImgMorphFinal",CV_WINDOW_NORMAL);
    imshow( "ImgMorphFinal", imgMorphFinal);
    cout <<"Type:" << imgMorphFinal.type();
    */

    // Mengembalikan warna dari img original yang memiliki 3 channel, convert image ke 3-ch img
    Mat imgKMeansColor(img.rows,img.cols,CV_8UC3);
    for(int y=0;y<img.rows;y++){
        for(int x=0;x<img.cols;x++){
            if(imgMorphOpen.at<float>(y,x)==255) // jika label nukleus
            {
                imgKMeansColor.at<Vec3b>(y,x)= img.at<Vec3b>(y,x); //diberi warna sesuai original
            }

           else
            {
                imgKMeansColor.at<Vec3b>(y,x) = {255,255,255}; //jika tidak diberi warna
            }
        }
    }

    // Show the result of img when assigned with original color
    namedWindow("Kmeans Original Color",WINDOW_NORMAL);
    imshow("Kmeans Original Color", imgKMeansColor);
    //imwrite("/home/reno/Pictures/WithoutMedian.jpg",imgKMeansColor);

    // Perform median filtering
    // Trying median filter 7x7
    Mat imgMedian;
    medianBlur(imgKMeansColor,imgMedian,7);
    // show result of median filtering 7x7
    namedWindow("Image Median",WINDOW_NORMAL);
    imshow("Image Median",imgMedian);

    // Perform Dist Transform
    // Create imgDistInput because distanceTransform() need a CV_8U for input img
    Mat imgDistInput;
    imgMorphOpen.convertTo(imgDistInput,CV_8UC3);
    Mat imgDistTransform;
    distanceTransform(imgDistInput, imgDistTransform, CV_DIST_L2, 5);
    // Normalize the distance image for range = {0.0, 1.0}
    // so we can visualize and threshold it
    normalize(imgDistTransform, imgDistTransform, 0, 1., NORM_MINMAX);

    namedWindow("Distant Transform",CV_WINDOW_NORMAL);
    imshow("Distant Transform", imgDistTransform);


    // Extract peaks for markers for foreground objects with threshold and dilation
    // Penentuan threshold ini akan mempengaruhi pembentukan marker dan akan mempengaruhi WT yang dihasilkan
    threshold(imgDistTransform, imgDistTransform, .4, 1., CV_THRESH_BINARY);
    namedWindow("Peaks",CV_WINDOW_NORMAL);
    imshow("Peaks", imgDistTransform);

    // Peaks with a little bit dilation
    //Mat kernel1 = Mat::ones(3, 3, CV_8UC1);
    //dilate(imgDistTransform, imgDistTransform, kernel1);
    //namedWindow("Peaks Dilate",CV_WINDOW_NORMAL);
    //imshow("Peaks Dilate", imgDistTransform);

    // ----------------------------------------------------------------------------------------------------------------------CEK SINI
     // Create markers for WT algorithm
     // Create the CV_8U version of the distance image
     // It is needed for findContours()
     Mat dist_8u;
     imgDistTransform.convertTo(dist_8u, CV_8U);

     // Find total markers
     vector<vector<Point> > contours;
     findContours(dist_8u, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

     // Create the marker image for the watershed algorithm
     Mat markers = Mat::zeros(imgDistTransform.size(), CV_32SC1); //SC1 -> Signed Char 1channel

     // Draw the foreground markers
     for (size_t i = 0; i < contours.size(); i++)
         drawContours(markers, contours, static_cast<int>(i), Scalar::all(static_cast<int>(i)+1), -1);

     // Draw the background marker
     circle(markers, Point(5,5), 3, CV_RGB(255,255,255), -1);
     namedWindow("Markers",CV_WINDOW_NORMAL);
     imshow("Markers", markers*10000);
     // -------------------------------------------------------------------------------------------------------------------SAMPAI SINI


    // Perform WT
    watershed(imgKMeansColor, markers);
    Mat mark = Mat::zeros(markers.size(), CV_8UC1);
    markers.convertTo(mark, CV_8UC1);
    bitwise_not(mark, mark);
    //imshow("Markers_v2", mark); // uncomment this if you want to see how the mark
                                   // image looks like at that point
    // Generate random colors
    vector<Vec3b> colors2;
    for (size_t i = 0; i < contours.size(); i++)
    {
        int b = theRNG().uniform(0, 255);
        int g = theRNG().uniform(0, 255);
        int r = theRNG().uniform(0, 255);
        colors2.push_back(Vec3b((uchar)b, (uchar)g, (uchar)r));
    }
    // Create the result image
    Mat dst = Mat::zeros(markers.size(), CV_8UC3);
    // Fill labeled objects with random colors
    for (int i = 0; i < img.rows; i++)
    {
        for (int j = 0; j < img.cols; j++)
        {
            int index = markers.at<int>(i,j);
            if (index > 0 && index <= static_cast<int>(contours.size()))
            {
                dst.at<Vec3b>(i,j) = colors[index-1];
            }
            else
            {
                dst.at<Vec3b>(i,j) = Vec3b(255,255,255);
            }
        }
    }

    // To check region
    cout << contours.size();
    Mat region1 = markers==7;
    namedWindow("Region1",CV_WINDOW_NORMAL);
    imshow("Region1", region1);

    // Visualize the final image
    namedWindow("Watershed",CV_WINDOW_NORMAL);
    imshow("Watershed", dst);

    // Extract binary nucleifrom WT
    Mat imgWT(img.rows,img.cols,CV_32F);
    for (int i = 0; i < img.rows; i++)
    {
        for (int j = 0; j < img.cols; j++)
        {
            int index = markers.at<int>(i,j);
            // Jika bukan boundary
            if (index > 0 && index <= static_cast<int>(contours.size()))
            {
                imgWT.at<float>(i,j)= 255; // beri sesuai warna
            }
            else
            {
                imgWT.at<float>(i,j) = 0; //jika tidak beri warna putih
            }
        }
    }
    namedWindow("Img WT Binary",CV_WINDOW_NORMAL);
    imshow("Img WT Binary", imgWT);

    // Extract colored nuclei from WT
    /*
    Mat imgWTFinal(img.rows,img.cols,CV_8UC3);
    for (int i = 0; i < img.rows; i++)
    {
        for (int j = 0; j < img.cols; j++)
        {
            int index = markers.at<int>(i,j);
            // Jika bukan boundary
            if (index > 0 && index <= static_cast<int>(contours.size()))
            {
                imgWTFinal.at<Vec3b>(i,j)= imgKMeansColor.at<Vec3b>(i,j); // beri sesuai warna
            }
            else
            {
                imgWTFinal.at<Vec3b>(i,j) = {255,255,255}; //jika tidak beri warna putih
            }
        }
    }
    namedWindow("Img WT Final",CV_WINDOW_NORMAL);
    imshow("Img WT Final", imgWTFinal);
    */
    waitKey();
    destroyAllWindows();
    return 0;
}
