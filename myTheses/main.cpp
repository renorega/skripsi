// ---------------------------- HEADER FILE -----------------------------------
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <cmath>
using namespace std;
using namespace cv;

// ----------------------------- FUNCTION DECLARATION------------------------------
Mat imgOriginal;
Mat preprocessing(Mat img);
Mat readImage(string path);
void showImage(string name,Mat img);
Mat segmentation(Mat img,int K);

class morphologicalOperationClass
{
private:
    Mat img, imgResult,imgResultColor;
    int kernelSize;
public:
    morphologicalOperationClass(Mat img,int kernelSize=7)
    {
        this->img = img;
        this->kernelSize = kernelSize;
    }

    void run();
    Mat getResult(){return imgResult;}
    Mat getResultColor(){return imgResultColor;}
};

class watershedClass
{
private:
    Mat img,imgColor, imgDistInput, imgDistTransform, dist_8u, markers,imgResult, imgResultColor;
    float thresholdValue;
    vector<vector<Point>> contours;

public:
    watershedClass(Mat img,float thresholdValue=0.4)
    {
        this->img = img;
        this->thresholdValue = thresholdValue;
    }

    void run();
    float getThreshold(){ return thresholdValue;}
    Mat getMarkers(){return markers;}
    vector<vector<Point>> getContours(){return contours;}
    Mat getResult(){return imgResult;}
    Mat getResultColor(){return imgResultColor;}

};

// ----------------------------------- MAIN FUNCTION--------------------------------------
int main()
{
    //1. READ IMAGE
    string location = "/home/reno/skripsi/ALL_SAMPLES/ALL_Sardjito/gambar_29mei/AfarelAzis_17april_01680124/";
    string nameFile= "8-9.jpg";
    cout <<"File: " << nameFile << endl;
    imgOriginal = readImage(location+nameFile);
    showImage("Original",imgOriginal);
    cout << "img rows : " <<  imgOriginal.rows << " img cols :" << imgOriginal.cols << endl;

    //2. PERFORM PREPROCESSING
    Mat imgPreprocessing= preprocessing(imgOriginal);

    //3. PERFORM SEGMENTATION
    int K = 4;
    Mat imgKMeans= segmentation(imgPreprocessing,K);
    showImage("Kmeans Binary",imgKMeans);

    //4. PERFORM MORPHOLOGICAL OPENING AND CLOSING
    morphologicalOperationClass morph(imgKMeans);
    morph.run();
    Mat imgMorph = morph.getResult();
    Mat imgMorphColor = morph.getResultColor();

    //5. PERFORMING WATERSHED TRANSFORMATION
    watershedClass wt(imgMorph);
    wt.run();
    float thresholdValue = wt.getThreshold();
    vector<vector<Point> > contours = wt.getContours();
    Mat markers = wt.getMarkers();

    // Create the result image for colored version and binary version
    Mat imgWT = wt.getResultColor();
    Mat imgWTBinary = wt.getResult();

    // Show the result of WT
    showImage("First WT Result",imgWT);

    // CALCULATE THE AREA OF EVERY REGION IN MARKERS MANUALLY
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

    // Find connected nuclei area that are larger than 36.000
    // Manually find max value from array
    int thresholdNuclei = 36000;
    float areaNucleiIndividu = 20000; // based on the sample of 100 nucleis, mean value of the area is 19.607

    vector<int>indexConnectedNuclei; // contain index of connected nuclei
    vector<float>areaConnectedNuclei; // contain area of connected nuclei
    bool isSecondWT = false; // flag to save

    //Print area result
    cout <<"Calculate area manually from WT img! \nResult:" << endl;
    for(int i=0; i<=static_cast<int>(contours.size());i++)
        cout << "Area-" << i << ": " << area[i] << endl;

    // Find connected nuclei
    cout << "Contours size: " << contours.size() << endl;
    for(int i=0; i<=static_cast<int>(contours.size());i++)
        // If background are extracted into WT region, so region with more than 100000 pixels are ignored
        if(area[i]>thresholdNuclei && area[i]<100000)
            indexConnectedNuclei.push_back(i);

    // Calculate target region connected nuclei
    vector<float>targetRegionConnectedNuclei; // contain target region of connected nuclei
    for (int i:indexConnectedNuclei)
    {
        int tempTarget = round(area[i]/areaNucleiIndividu);
        targetRegionConnectedNuclei.push_back(tempTarget);
    }

    if(indexConnectedNuclei.size()>0)
        isSecondWT = true;

//    Mat test= markers== 7;
//    namedWindow("Test",CV_WINDOW_NORMAL);
//    imshow("Test", test);



    // PERFORM SECOND WT
    Mat connectedNucleiColor(imgOriginal.rows,imgOriginal.cols,CV_8UC3);
    Mat allConnectedNucleiPeaks;
    vector<vector<Point>>listContoursConnectedNuclei;
    Mat markersConnectedNuclei ;
            //Mat::zeros(allConnectedNucleiPeaks.size(), CV_32SC1); //SC1 -> Signed Char 1channel

    if(isSecondWT)
    {
        cout << "\nPerforming second WT . . .\n" << endl;

        // Print indexConnectedNuclei and its area
        for (int i=0;i<static_cast<int>(indexConnectedNuclei.size());i++)
            cout << "Index connected nuclei: " << indexConnectedNuclei[i] << " Area connected nuclei : " << area[indexConnectedNuclei[i]] << " Target region: " << targetRegionConnectedNuclei[i] << endl;

        // Extract the connected nuclei
        Mat connectedNuclei;
        for(int i=0;i<static_cast<int>(indexConnectedNuclei.size());i++)
            connectedNuclei = connectedNuclei + (markers==indexConnectedNuclei[i]);

        showImage("Connected Nuclei",connectedNuclei);

        // Colored result of Connected Nuclei
        Mat connectedNucleiColor(imgOriginal.rows,imgOriginal.cols,CV_8UC3);

        for(int y=0;y<imgOriginal.rows;y++)
        {
            for(int x=0;x<imgOriginal.cols;x++)
            {
                if(connectedNuclei.at<unsigned char>(y,x)) // jika ada warna atau tidak sama dengan 0
                    connectedNucleiColor.at<Vec3b>(y,x)= imgMorphColor.at<Vec3b>(y,x); //diberi warna sesuai original

                else
                    connectedNucleiColor.at<Vec3b>(y,x) = Vec3b{255,255,255}; //jika tidak diberi warna
            }
        }

        for(int i=0;i<static_cast<int>(indexConnectedNuclei.size());i++)
        {
            cout <<"Performing the " << i << " index"<<endl;
            int targetContour = targetRegionConnectedNuclei[i];
            cout <<"Target contour of-" <<i<<":" << targetContour << endl;
            Mat connectedNuclei= markers== indexConnectedNuclei[i];

            // Perform Dist Transform
            // Create imgDistInput because distanceTransform() need a CV_8U for input img
            Mat connectedNucleiDistInput;
            connectedNuclei.convertTo(connectedNucleiDistInput,CV_8U);
            Mat connectedNucleiDistTransform;
            distanceTransform(connectedNucleiDistInput, connectedNucleiDistTransform, CV_DIST_L2, 5);

            normalize(connectedNucleiDistTransform, connectedNucleiDistTransform, 0, 1., NORM_MINMAX);

            thresholdValue += 0.05; // This value for thresholding second WT
            Mat connectedNucleiPeaks;
            threshold(connectedNucleiDistTransform, connectedNucleiPeaks, thresholdValue, 1., CV_THRESH_BINARY);

            // Create markers for WT algorithm
            // Create the CV_8U version of the distance image
            // It is needed for findContours()
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
                listContoursConnectedNuclei.insert(listContoursConnectedNuclei.end(),contoursConnectedNuclei.begin(),contoursConnectedNuclei.end());

            while(static_cast<int>(contoursConnectedNuclei.size())!=targetContour)
            {
                // If threshold value reached 1.0 or more and can't get target Contour
                // To prevent from infinity loop
                if(thresholdValue>=1.0) break;

                // If contours size is larger than targetContour but smaller than largerSize
                if(static_cast<int>(indexConnectedNuclei.size())>targetContour && static_cast<int>(indexConnectedNuclei.size())<largerSize && !isLargerSize)
                {
                    //contoursConnectedNucleiLarger = contoursConnectedNuclei;
                    //connectedNucleiPeaksLarger = connectedNucleiPeaks;
                    isLargerSize = true;
                    largerSize = contoursConnectedNuclei.size();
                    thresholdValueBig = thresholdValue;
                }

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

            if(!successWT && !isLargerSize)
                cout << "This is single nucleus!" << endl;

            // END OF NEW SECOND WT
            allConnectedNucleiPeaks = allConnectedNucleiPeaks + (connectedNucleiPeaks!=0);
        }

        markersConnectedNuclei = Mat::zeros(allConnectedNucleiPeaks.size(), CV_32SC1); //SC1 -> Signed Char 1channel

        // Draw the foreground markers
          for (size_t i = 0; i < listContoursConnectedNuclei.size(); i++)
                drawContours(markersConnectedNuclei, listContoursConnectedNuclei, static_cast<int>(i), Scalar::all(static_cast<int>(i)+1), -1);

        // Show markers result
        //showImage("Markers Second WT", markersConnectedNuclei*10000);

        // Draw the background marker
        circle(markersConnectedNuclei, Point(5,5), 3, CV_RGB(255,255,255), -1);
        watershed(connectedNucleiColor,markersConnectedNuclei);

        for (int i = 0; i < imgOriginal.rows; i++)
        {
            for (int j = 0; j < imgOriginal.cols; j++)
            {
                int index = markersConnectedNuclei.at<int>(i,j);
                if(index<0)
                {
                    imgWT.at<Vec3b>(i,j) =Vec3b(255,255,255);
                    imgWTBinary.at<char>(i,j) = 0;
                }
            }
        }

        showImage("Final WT Result", imgWTBinary);
    }

    // Fill labeled objects with random colors and make binary image

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

Mat readImage(string path)
{
    Mat img = imread(path,IMREAD_COLOR);
    return img;
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

Mat segmentation(Mat img,int K)
{
    vector<Mat> imgK;
    split(img, imgK);
    //PERFORM KMEANS
    Mat p = Mat::zeros(img.cols*img.rows, 1, CV_32F);
    Mat bestLabels,centers;
    for(int i=0; i<img.cols*img.rows; i++)
        p.at<float>(i,0) = imgK[1].data[i] / 255.0; // Use only one channel(S) with normalization to ease computation
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
    Mat imgResult= Mat(img.rows, img.cols, CV_32F);
    for(int i=0; i<img.cols*img.rows; i++) {
        imgResult.at<float>(i/img.cols, i%img.cols) = (float)(colors[bestLabels.at<int>(0,i)]);
    }
    return imgResult;
}

void morphologicalOperationClass::run()
{
    Mat element = getStructuringElement( MORPH_RECT, Size( kernelSize,kernelSize));
    morphologyEx(img, imgResult, MORPH_OPEN, element );
    morphologyEx( imgResult, imgResult, MORPH_CLOSE, element );
    imgResultColor = Mat(imgOriginal.rows,imgOriginal.cols,CV_8UC3);
    for(int y=0;y<imgOriginal.rows;y++)
        for(int x=0;x<imgOriginal.cols;x++)
        {
            if(imgResult.at<float>(y,x)==255) // jika label nukleus
                imgResultColor.at<Vec3b>(y,x)= imgOriginal.at<Vec3b>(y,x); //diberi warna sesuai original
            else
                imgResultColor.at<Vec3b>(y,x) = {255,255,255}; //jika tidak diberi warna
        }
}

void watershedClass::run()
{
    imgColor = Mat(img.rows,img.cols,CV_8UC3);
    for(int y=0;y<imgOriginal.rows;y++){
        for(int x=0;x<imgOriginal.cols;x++){
            if(img.at<float>(y,x)==255) // jika label nukleus
                imgColor.at<Vec3b>(y,x)= imgOriginal.at<Vec3b>(y,x); //diberi warna sesuai original

           else
                imgColor.at<Vec3b>(y,x) = {255,255,255}; //jika tidak diberi warna
        }
    }

    img.convertTo(imgDistInput,CV_8UC3);
    distanceTransform(imgDistInput, imgDistTransform, CV_DIST_L2, 5);
    normalize(imgDistTransform, imgDistTransform, 0, 1., NORM_MINMAX);
    threshold(imgDistTransform, imgDistTransform, thresholdValue, 1., CV_THRESH_BINARY);
    imgDistTransform.convertTo(dist_8u, CV_8U);
    findContours(dist_8u, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
    markers = Mat::zeros(imgDistTransform.size(), CV_32SC1); //SC1 -> Signed Char 1channel
    for (size_t i = 0; i < contours.size(); i++)
        drawContours(markers, contours, static_cast<int>(i), Scalar::all(static_cast<int>(i)+1), -1);

    // Draw the background marker
    circle(markers, Point(5,5), 3, CV_RGB(255,255,255), -1);

    // Perform First WT
    watershed(imgColor, markers);

    imgResultColor = Mat(img.rows,img.cols,CV_8UC3);
    imgResult= Mat::zeros(img.size(), CV_8UC1); // CV_8UC3
    for (int i = 0; i < imgOriginal.rows; i++)
    {
        for (int j = 0; j < imgOriginal.cols; j++)
        {
            int index = markers.at<int>(i,j);
            if (index > 0 && index <= static_cast<int>(contours.size()))
                imgResult.at<char>(i,j) = 255;
            else
                imgResult.at<char>(i,j) = 0;
        }
    }

    for (int i = 0; i < imgOriginal.rows; i++)
    {
        for (int j = 0; j < imgOriginal.cols; j++)
        {
            if (imgResult.at<char>(i,j))
                imgResultColor.at<Vec3b>(i,j) = imgColor.at<Vec3b>(i,j) ;
            else
                imgResultColor.at<Vec3b>(i,j) =Vec3b(255,255,255);
        }
    }

}
