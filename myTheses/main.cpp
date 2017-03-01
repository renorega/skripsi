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
class segmentationClass
{
private:
    Mat img,imgResult,imgResultColor,bestLabels,centers;
    vector<Mat> imgK;
    vector<float> sat;
    float highestSatValue;
    int K, highestSatIndex;
public:
    segmentationClass(Mat img,int K=4)
    {
        this->img = img;
        this->K = K;
    }
    void run();
    void findHighestSaturation();
    void assignColor();
    Mat getResult(){return imgResult;}
    Mat getResultColor(){return imgResultColor;}
};
class watershedClass
{
private:
    Mat img,imgColor,imgResult,imgResultColor,markers;
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
    watershedClass(Mat img,float thresholdValue=0.4)
    {
        this->img = img;
        this->thresholdValue = thresholdValue;
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

// ----------------------------------- MAIN FUNCTION--------------------------------------
int main()
{
    //1. READ IMAGE
    string location = "/home/reno/skripsi/ALL_SAMPLES/ALL_Sardjito/gambar_29mei/AfarelAzis_17april_01680124/";
    string nameFile= "103-107.jpg";
    cout <<"File: " << nameFile << endl;
    imgOriginal = readImage(location+nameFile);
    showImage("Original",imgOriginal);
    cout << "img rows : " <<  imgOriginal.rows << " img cols :" << imgOriginal.cols << endl;

    //2. PERFORM PREPROCESSING
    Mat imgPreprocessing= preprocessing(imgOriginal);

    //3. PERFORM SEGMENTATION
    segmentationClass segment(imgPreprocessing);
    segment.run();
    Mat imgKMeans = segment.getResult();
    showImage("Kmeans Binary",imgKMeans);

    //4. PERFORM MORPHOLOGICAL OPENING AND CLOSING
    morphologicalOperationClass morph(imgKMeans);
    morph.run();

    //5. PERFORMING WATERSHED TRANSFORMATION
    watershedClass wt(morph.getResult());
    wt.run();
    showImage("First WT Result",wt.getResultColor());
    wt.calculateArea();

    wt.setThresholdNuclei(36000);
    wt.setAreaNucleiIndividu(20000);
    wt.findConnectedNuclei();
    if(wt.getIsSecondWT())
        showImage("Final WT Result",wt.getResultColor());
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

    Mat imgDistInput, imgDistTransform, dist_8u;

    img.convertTo(imgDistInput,CV_8UC3);
    distanceTransform(imgDistInput, imgDistTransform, CV_DIST_L2, 5);
    normalize(imgDistTransform, imgDistTransform, 0, 1., NORM_MINMAX);
    threshold(imgDistTransform, imgDistTransform, thresholdValue, 1., CV_THRESH_BINARY);
    imgDistTransform.convertTo(dist_8u, CV_8U);
    findContours(dist_8u, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
    markers = Mat::zeros(imgDistTransform.size(), CV_32SC1); //SC1 -> Signed Char 1channel
    for (size_t i = 0; i < contours.size(); i++)
        drawContours(markers, contours, static_cast<int>(i), Scalar::all(static_cast<int>(i)+1), -1);

    circle(markers, Point(5,5), 3, CV_RGB(255,255,255), -1);

    watershed(imgColor, markers);

    imgResultColor = Mat(img.rows,img.cols,CV_8UC3);
    imgResult= Mat::zeros(img.size(), CV_8UC1);
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
        for(int y=0;y<imgOriginal.rows;y++)
            for(int x=0;x<imgOriginal.cols;x++)
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

        showImage("Connected Nuclei",connectedNuclei);
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

    connectedNucleiColor = Mat(imgColor.rows,imgColor.cols,CV_8UC3);
    for(int y=0;y<imgOriginal.rows;y++)
        for(int x=0;x<imgOriginal.cols;x++)
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

        thresholdValue += 0.05; // This value for thresholding second WT
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
            listContoursConnectedNuclei.insert(listContoursConnectedNuclei.end(),contoursConnectedNuclei.begin(),contoursConnectedNuclei.end());

        while(static_cast<int>(contoursConnectedNuclei.size())!=targetContour)
        {
            if(thresholdValue>=1.0) break;

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

        if(!successWT && !isLargerSize)
            cout << "This is single nucleus!" << endl;

        // END OF NEW SECOND WT
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

    for (int i = 0; i < imgOriginal.rows; i++)
        for (int j = 0; j < imgOriginal.cols; j++)
        {
            int index = markersConnectedNuclei.at<int>(i,j);
            if(index<0)
            {
                imgResultColor.at<Vec3b>(i,j) =Vec3b(255,255,255);
                imgResult.at<char>(i,j) = 0;
            }
        }
}
