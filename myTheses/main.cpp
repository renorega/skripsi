#include <opencv2/opencv.hpp>
#include <iostream>
using namespace std;
using namespace cv;

int main(int argc, char *argv[])
{
    //loading image
    Mat img,imgHSV;

    // use IMREAD_COLOR to access image in BGR format as 8 bit image
    img = imread("/home/reno/skripsi/ALL_SAMPLES/ALL_Sardjito/gambar_29mei/AfarelAzis_17april_01680124/29-39.jpg",IMREAD_COLOR);
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
    Mat imgMedian;
    medianBlur(imgEqualized,imgMedian,7);
    /*
    // show result of median filtering 7x7
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
    split(imgMedian, imgK);

    /*
    // show S-channel from images
    namedWindow("S",WINDOW_NORMAL);
    imshow("S", imgK[1]);
    */

    //Aplikasikan algoritma KMEANS
    //buat matrix P kosong dengan ukuran row = column x row, col=1 dan tipe CV_32F
    Mat p = Mat::zeros(img.cols*img.rows, 1, CV_32F);
    Mat bestLabels,centers,clustered;
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
        if(i==indexSat) colors[i]=1; // jika label nukleus diberi warna putih
        else colors[i]=0;
    }

    // Ngereshape plus ngasih warna dengan best labels
    clustered = Mat(img.rows, img.cols, CV_32F);
    for(int i=0; i<img.cols*img.rows; i++) {
        clustered.at<float>(i/img.cols, i%img.cols) = (float)(colors[bestLabels.at<int>(0,i)]);
    }


    // show the result of img with binary color
    namedWindow("clustered",WINDOW_NORMAL);
    imshow("clustered", clustered);

    // Perform Morphological Opening
    // varible destination
    Mat imgMorphOpen,imgMorphClose, imgMorphFinal;
    // kernel size
    int kernelSize = 7;
    // creating structured element for morphological operation
    Mat element = getStructuringElement( MORPH_RECT, Size( kernelSize,kernelSize));
    // perform morphological opening
    morphologyEx( clustered, imgMorphOpen, MORPH_OPEN, element );
    namedWindow("ImgMorphOpen",CV_WINDOW_NORMAL);
    imshow( "ImgMorphOpen", imgMorphOpen);

    // Perform Morphological Closing
    kernelSize = 7;
    element = getStructuringElement( MORPH_RECT, Size( kernelSize,kernelSize));
    morphologyEx( clustered, imgMorphClose, MORPH_CLOSE, element );
    namedWindow("ImgMorphClose",CV_WINDOW_NORMAL);
    imshow( "ImgMorphClose", imgMorphClose);

    // Perform Morphological Opening + Closing
    morphologyEx( imgMorphOpen, imgMorphFinal, MORPH_CLOSE, element );
    namedWindow("ImgMorphFinal",CV_WINDOW_NORMAL);
    imshow( "ImgMorphFinal", imgMorphFinal);

    /*
    int const max_operator = 4;
    int const max_elem = 2;
    int const max_kernel_size = 21;
    createTrackbar("Operator:\n 0: Opening - 1: Closing \n 2: Gradient - 3: Top Hat \n 4: Black Hat", window_name, &morph_operator, max_operator, Morphology_Operations );

    /// Create Trackbar to select kernel type
    createTrackbar( "Element:\n 0: Rect - 1: Cross - 2: Ellipse", window_name,
                    &morph_elem, max_elem,
                    Morphology_Operations );

    /// Create Trackbar to choose kernel size
    createTrackbar( "Kernel size:\n 2n +1", window_name,
                    &morph_size, max_kernel_size,
                    Morphology_Operations );
    */

    /*
    // Mengembalikan warna dari img original yang memiliki 3 channel, convert image ke 3-ch img
    Mat imgKFinal(img.rows,img.cols,CV_8UC3);
    for(int y=0;y<img.rows;y++){
        for(int x=0;x<img.cols;x++){
            if(clustered.at<float>(y,x)==1) // jika label nukleus
            {
                imgKFinal.at<Vec3b>(y,x)= img.at<Vec3b>(y,x); //diberi warna sesuai original
            }

           else
            {
                imgKFinal.at<Vec3b>(y,x) = {255,255,255}; //jika tidak diberi warna putih
            }
        }
    }

    // Show the result of img when assigned with original color
    namedWindow("Kmeans",WINDOW_NORMAL);
    imshow("Kmeans", imgKFinal);
    */
    waitKey();
    destroyAllWindows();
    return 0;
}
