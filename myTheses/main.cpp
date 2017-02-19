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

    //PERFORM KMEANS
    //Make empty Mat P with row = column x row, col=1 and type = CV_32F
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

    // Calculate mean value of saturation for every kmeans labels
    float sat[K] = {0}; // store mean value of saturation
    int div[K] = {0}; // store total area of every labels
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
    float maxSat = sat[0]; // store highest saturation value
    int indexSat = 0; // store labels with highest saturation
    for(int i=1;i<K;i++)
    {
        if(maxSat<sat[i])
        {
            maxSat = sat[i];
            indexSat = i;
        }
    }

    // Assign labels with white
    int colors[K]; //0 is black, 255 is white
    for(int i=0;i<K;i++)
    {
        if(i==indexSat) colors[i]=255;
        else colors[i]=0;
    }

    // Reshape and give color
    imgKMeansGrayscale= Mat(img.rows, img.cols, CV_32F);
    for(int i=0; i<img.cols*img.rows; i++) {
        imgKMeansGrayscale.at<float>(i/img.cols, i%img.cols) = (float)(colors[bestLabels.at<int>(0,i)]);
    }
    /*
    // Show the result of KMeans with grayscale
    namedWindow("Grayscale KMeans",WINDOW_NORMAL);
    imshow("Grayscale KMeans", imgKMeansGrayscale);
    */

    // Perform Morphological Opening
    Mat imgMorphOpen;
    int kernelSize = 7;
    Mat element = getStructuringElement( MORPH_RECT, Size( kernelSize,kernelSize));
    morphologyEx(imgKMeansGrayscale, imgMorphOpen, MORPH_OPEN, element );

    namedWindow("ImgMorphOpen",CV_WINDOW_NORMAL);
    imshow( "ImgMorphOpen", imgMorphOpen);

    // Perform Morphological Closing
    kernelSize = 7;
    element = getStructuringElement( MORPH_RECT, Size( kernelSize,kernelSize));
    /*
    morphologyEx( clustered, imgMorphClose, MORPH_CLOSE, element );
    namedWindow("ImgMorphClose",CV_WINDOW_NORMAL);
    imshow( "ImgMorphClose", imgMorphClose);
    */

    // Perform Morphological Opening + Closing
    morphologyEx( imgMorphOpen, imgMorphOpen, MORPH_CLOSE, element );
    namedWindow("ImgMorphFinal",CV_WINDOW_NORMAL);
    imshow( "ImgMorphFinal", imgMorphOpen);

    // Colored result of KMeans
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

    //imwrite("/home/reno/Pictures/WithoutMedian.jpg",imgKMeansColor);

    // Perform median filtering
    // Trying median filter 7x7
    Mat imgMedian;
    medianBlur(imgKMeansColor,imgMedian,7);
    /*
    // show result of median filtering 7x7
    namedWindow("Image Median",WINDOW_NORMAL);
    imshow("Image Median",imgMedian);
    */

    // Show the result of img when assigned with original color
    namedWindow("Kmeans Result Colored",WINDOW_NORMAL);
    imshow("Kmeans Result Colored", imgKMeansColor);

    // Perform Dist Transform
    // Create imgDistInput because distanceTransform() need a CV_8U for input img
    Mat imgDistInput;
    imgMorphOpen.convertTo(imgDistInput,CV_8UC3);
    Mat imgDistTransform;
    distanceTransform(imgDistInput, imgDistTransform, CV_DIST_L2, 5);
    // Normalize the distance image for range = {0.0, 1.0}
    // so we can visualize and threshold it
    normalize(imgDistTransform, imgDistTransform, 0, 1., NORM_MINMAX);

    /*
    namedWindow("Distant Transform",CV_WINDOW_NORMAL);
    imshow("Distant Transform", imgDistTransform);
    */

    // Extract peaks for markers for foreground objects with threshold and dilation
    // Penentuan threshold ini akan mempengaruhi pembentukan marker dan akan mempengaruhi WT yang dihasilkan
    threshold(imgDistTransform, imgDistTransform, .4, 1., CV_THRESH_BINARY);
    /*
    namedWindow("Peaks",CV_WINDOW_NORMAL);
    imshow("Peaks", imgDistTransform);
    */

    // Peaks with a little bit dilation
    //Mat kernel1 = Mat::ones(3, 3, CV_8UC1);
    //dilate(imgDistTransform, imgDistTransform, kernel1);
    //namedWindow("Peaks Dilate",CV_WINDOW_NORMAL);
    //imshow("Peaks Dilate", imgDistTransform);

     // Create markers for WT algorithm
     // Create the CV_8U version of the distance image
     // It is needed for findContours()
     Mat dist_8u;
     imgDistTransform.convertTo(dist_8u, CV_8U);

     // Show img dist_8u for WT
     /*
     namedWindow("dist8u",CV_WINDOW_NORMAL);
     imshow("dist8u", dist_8u);
     */

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

     /*
     // Show markers result
     namedWindow("Markers",CV_WINDOW_NORMAL);
     imshow("Markers", markers*10000);
     */


    // Perform WT
    watershed(imgKMeansColor, markers);
    // Generate random colors
    vector<Vec3b> colors2;
    for (size_t i = 0; i < contours.size(); i++)
    {
        int b = theRNG().uniform(0, 255);
        int g = theRNG().uniform(0, 255);
        int r = theRNG().uniform(0, 255);
        colors2.push_back(Vec3b((uchar)b, (uchar)g, (uchar)r));
    }
    // Create the result image for colored version and binary version
    Mat imgWT = Mat::zeros(markers.size(), CV_8UC3); // CV_8UC3
    Mat imgWTBinary = Mat::zeros(markers.size(), CV_8UC1); // CV_8UC3

    // Fill labeled objects with random colors and make binary image
    for (int i = 0; i < img.rows; i++)
    {
        for (int j = 0; j < img.cols; j++)
        {
            int index = markers.at<int>(i,j);
            if (index > 0 && index <= static_cast<int>(contours.size()))
            {
                imgWT.at<Vec3b>(i,j) = colors2[index-1];
                imgWTBinary.at<char>(i,j) = 255;
            }
            else
            {
                imgWT.at<Vec3b>(i,j) =Vec3b(255,255,255);
                imgWTBinary.at<char>(i,j) = 0;
            }
        }
    }

    // Show the result of WT
    namedWindow("WT Result",CV_WINDOW_NORMAL);
    imshow("WT Result", imgWT);

    // Show the binary result of WT Binary
    namedWindow("Binary Watershed",CV_WINDOW_NORMAL);
    imshow("Binary Watershed", imgWTBinary);

    // Calculate the area of every markers manually
    float area[contours.size()+1] = {0};
    for (int row=0;row<markers.rows;row++)
    {
        for(int col=0;col<markers.cols;col++)
        {
            // because background are not included in contours , so we can use <contours.size()+1 or <=contours.size()
            for(int index=0;index<=static_cast<int>(contours.size());index++)
            {
                if(markers.at<int>(row,col)==index)
                    area[index]++;
            }
        }
    }

    // Manually find max value from array
    float maxArea=0;
    int indexMaxArea;
    cout << "Contours size: " << contours.size() << endl;
    for(int i=0; i<=static_cast<int>(contours.size());i++)
        // If background are extracted into WT region, so region with more than 100000 pixels are ignored
        if(area[i]>maxArea && area[i]<100000)
        {
            maxArea=area[i];
            indexMaxArea = i;
        }

    //Print area result
    cout <<"Calculate area manually from WT img! \nResult:" << endl;
    for(int i=0; i<=static_cast<int>(contours.size());i++)
        cout << "Area-" << i << ": " << area[i] << endl;

    // Print max area
    cout << "Max area of WT: " << maxArea << " from index: " << indexMaxArea <<endl;

    // Find region that are larger than 40.000 pixels
    // PERFORM SECOND WT IF THERE IS STILL AREA LARGER THAN 40.000 pixels!
    float indexConnectedNuclei;
    if(maxArea>40000)
    {
        indexConnectedNuclei = indexMaxArea;
        // Show connected nuclei from WT
        Mat connectedNuclei= markers==indexConnectedNuclei;
        namedWindow("Connected Nuclei",CV_WINDOW_NORMAL);
        imshow("Connected Nuclei", connectedNuclei);

        // Perform second watershed
        // Perform Dist Transform
        // Create imgDistInput because distanceTransform() need a CV_8U for input img
        connectedNuclei.convertTo(connectedNuclei,CV_8UC1);
        Mat connectedNucleiDistTransform;
        distanceTransform(connectedNuclei, connectedNucleiDistTransform, CV_DIST_L2, 5);
        // Normalize the distance image for range = {0.0, 1.0}
        // so we can visualize and threshold it
        normalize(connectedNucleiDistTransform, connectedNucleiDistTransform, 0, 1., NORM_MINMAX);

        // Show distant transform of connected nuclei
        namedWindow("Connected Nuclei Distant Transform",CV_WINDOW_NORMAL);
        imshow("Connected Nuclei Distant Transform", connectedNucleiDistTransform);

        // Extract peaks for markers for foreground objects with threshold and dilation
        // Penentuan threshold ini akan mempengaruhi pembentukan marker dan akan mempengaruhi WT yang dihasilkan
        threshold(connectedNucleiDistTransform, connectedNucleiDistTransform, .8, 1., CV_THRESH_BINARY);

        // Show peaks of connected nuclei
        namedWindow("Peaks Connected Nuclei",CV_WINDOW_NORMAL);
        imshow("Peaks Connected Nuclei", connectedNucleiDistTransform);

        // Peaks with a little bit dilation
        //Mat kernel1 = Mat::ones(3, 3, CV_8UC1);
        //dilate(imgDistTransform, imgDistTransform, kernel1);
        //namedWindow("Peaks Dilate",CV_WINDOW_NORMAL);
        //imshow("Peaks Dilate", imgDistTransform);

         // Create markers for WT algorithm
         // Create the CV_8U version of the distance image
         // It is needed for findContours()
         Mat connectedNuclei8u;
         connectedNucleiDistTransform.convertTo(connectedNuclei8u, CV_8U);

         // Show img dist_8u for WT
         /*
         namedWindow("dist8u",CV_WINDOW_NORMAL);
         imshow("dist8u", dist_8u);
         */

         // Find total markers
         vector<vector<Point> > contoursConnectedNuclei;
         findContours(connectedNuclei8u, contoursConnectedNuclei, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
         /*
         if(contoursConnectedNuclei.size()<2)
         {
             conne
         }
         */

         // Create the marker image for the watershed algorithm
         Mat markersConnectedNuclei = Mat::zeros(imgDistTransform.size(), CV_32SC1); //SC1 -> Signed Char 1channel

         // Draw the foreground markers
         for (size_t i = 0; i < contoursConnectedNuclei.size(); i++)
             drawContours(markersConnectedNuclei, contoursConnectedNuclei, static_cast<int>(i), Scalar::all(static_cast<int>(i)+1), -1);

         cout << "Contours size of connected nuclei : " << contoursConnectedNuclei.size() << endl;
         // Draw the background marker
         circle(markersConnectedNuclei, Point(5,5), 3, CV_RGB(255,255,255), -1);
         /*
         // Show markers result
         namedWindow("Markers",CV_WINDOW_NORMAL);
         imshow("Markers", markers*10000);
         */

         // Colored result of Connected Nuclei
         Mat connectedNucleiColor(img.rows,img.cols,CV_8UC3);
         for(int y=0;y<img.rows;y++)
         {
             for(int x=0;x<img.cols;x++)
             {
                 if(connectedNuclei.at<unsigned char>(y,x)) // jika ada warna atau tidak sama dengan 0
                 {
                     connectedNucleiColor.at<Vec3b>(y,x)= imgKMeansColor.at<Vec3b>(y,x); //diberi warna sesuai original
                 }

                else
                 {
                     connectedNucleiColor.at<Vec3b>(y,x) = Vec3b{255,255,255}; //jika tidak diberi warna
                 }
             }
         }

         // Show connected nuclei color
         namedWindow("Connected Nuclei Color",CV_WINDOW_NORMAL);
         imshow("Connected Nuclei Color",connectedNucleiColor );

        // Perform WT
        watershed(connectedNucleiColor, markersConnectedNuclei);

        // Create the result image for colored version and binary version
        Mat imgWTConnectedNuclei = Mat::zeros(markersConnectedNuclei.size(), CV_8UC3); // CV_8UC3
        Mat imgWTBinaryConnectedNuclei = Mat::zeros(markersConnectedNuclei.size(), CV_8UC1); // CV_8UC3
        // Fill labeled objects with random colors and make binary image
        for (int i = 0; i < img.rows; i++)
        {
            for (int j = 0; j < img.cols; j++)
            {
                int index = markersConnectedNuclei.at<int>(i,j);
                if (index > 0 && index <= static_cast<int>(contoursConnectedNuclei.size()))
                {
                    imgWTConnectedNuclei.at<Vec3b>(i,j) = colors2[index-1];
                    imgWTBinaryConnectedNuclei.at<char>(i,j) = 255;
                }
                else
                {
                    imgWTConnectedNuclei.at<Vec3b>(i,j) =Vec3b(255,255,255);
                    imgWTBinaryConnectedNuclei.at<char>(i,j) = 0;
                }
            }
        }

        // Show the result of WT
        namedWindow("WT Result of Connected Nuclei",CV_WINDOW_NORMAL);
        imshow("WT Result of Connected Nuclei", imgWTConnectedNuclei);

        for (int i = 0; i < img.rows; i++)
        {
            for (int j = 0; j < img.cols; j++)
            {
                if (markersConnectedNuclei.at<signed int>(i,j)<0)
                {
                    imgWTBinary.at<char>(i,j) = 0;
                }
            }
        }
        namedWindow("Merge Second WT Result",CV_WINDOW_NORMAL);
        imshow("Merge Second WT Result",imgWTBinary);
    }

    /*
    // Calculate area inside every contour
    for (unsigned int i = 0;  i < contours.size();  i++)
    {
        // Output contour points
        //cout << "# of contour points: " << contours[i].size() << std::endl;

         // Output every point of contour
         for (unsigned int j=0;  j<contours[i].size();  j++)
         {
             cout << "Point(x,y)=" << contours[i][j] << std::endl;
         }
        // Calculate the area inside contours
        cout << " Area: " << i <<": "<< contourArea(contours[i]) << std::endl;
    }
    */

    /*
    // These lines will find the contour from binary image with findContours and use contourArea to get the area of every contour
    // BUG!!!
    imgWTBinary.convertTo(imgWTBinary, CV_8U);
    vector<vector<Point> > contours_WT;
    findContours(imgWTBinary, contours_WT, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

    cout << "WT area" << endl;
    // Calculate area inside every contour
    for (unsigned int i = 0;  i < contours_WT.size();  i++)
    {
        // Output contour points
        //cout << "# of contour points: " << contours[i].size() << std::endl;

         /*
         // Output every point of contour
         for (unsigned int j=0;  j<contours[i].size();  j++)
         {
             cout << "Point(x,y)=" << contours[i][j] << std::endl;
         }
        // Calculate the area inside contours
        cout << " Area: " << i <<": "<< contourArea(contours_WT[i]) << std::endl;
    }
    */

    /*
    // These lines of code will try to find the area of blobs
    // Setup SimpleBlobDetector parameters.
    SimpleBlobDetector::Params params;

    // Change thresholds
    params.minThreshold = 10;
    params.maxThreshold = 200;

    // Filter by Area.
    params.filterByArea = true;
    params.minArea = 1500;

    // Filter by Circularity
    params.filterByCircularity = true;
    params.minCircularity = 0.1;

    // Filter by Convexity
    params.filterByConvexity = true;
    params.minConvexity = 0.87;

    // Filter by Inertia
    params.filterByInertia = true;
    params.minInertiaRatio = 0.01;

    #if CV_MAJOR_VERSION < 3   // If you are using OpenCV 2

      // Set up detector with params
      SimpleBlobDetector detector(params);

      // You can use the detector this way
      // detector.detect( im, keypoints);

    #else

      // Set up detector with params
      Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);

      // SimpleBlobDetector::create creates a smart pointer.
      // So you need to use arrow ( ->) instead of dot ( . )
      // detector->detect( im, keypoints);

    #endif
    */

    // Set up the detector with default parameters.
    /*
    SimpleBlobDetector detector(params);
    // Detect blobs.
    std::vector<KeyPoint> keypoints;
    detector.detect( imgWT, keypoints);
    // Draw detected blobs as red circles.
    // DrawMatchesFlags::DRAW_RICH_KEYPOINTS flag ensures the size of the circle corresponds to the size of blob
    Mat im_with_keypoints;
    drawKeypoints( imgWT, keypoints, im_with_keypoints, Scalar(0,0,255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );

    // Show blobs
    namedWindow("keypoints",CV_WINDOW_NORMAL);
    imshow("keypoints", im_with_keypoints );
    */

    // Extract colored nuclei from WT
    Mat imgWTFinal(img.rows,img.cols,CV_8UC3);
    for (int i = 0; i < img.rows; i++)
    {
        for (int j = 0; j < img.cols; j++)
        {
            if (imgWTBinary.at<char>(i,j))
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

    waitKey();
    destroyAllWindows();
    return 0;
}
