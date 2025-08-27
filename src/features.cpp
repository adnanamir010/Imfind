/*
 * CS 5330: Pattern Recognition And Computer Vision
 * Spring 2024
 * Project 2: Content Based Image Recognition
 * Authors: Adnan Amir & Tenzin Kunsang
 * Purpose of file: Holds all the functions used to generate feature vectors
 */
#include "opencv2/opencv.hpp"
#include <vector>

using namespace cv;
using namespace std;

//Function to get features from a 7x7 roi in the center
int classic(Mat &img, vector<float> &features){
    // Define a region of interest at the center of the image
    int center_x = img.cols / 2;
    int center_y = img.rows / 2;
    Rect roi(center_x - 3, center_y - 3, 7, 7);
    Mat feature_mat = img(roi).clone();

    // Extract pixel values from the region of interest
    for (int i = 0; i < feature_mat.rows; i++) {
        for (int j = 0; j < feature_mat.cols; j++) {
            Vec3b pixel = feature_mat.at<Vec3b>(i, j); // Assuming a 3-channel (BGR) image
            for (int k = 0; k < 3; k++) {
                features.push_back(static_cast<float>(pixel[k]));
            }
        }
    }
    return 0;
}

// Function to compute 3D histogram features from an image
int histogram3dfeatures(Mat &img, const int bins, vector<float> &features){
    // Clear any existing features and resize the features vector to accommodate the histogram
    features.clear();
    features.resize(bins*bins*bins, 0);

    for (int i = 0; i < img.rows; i++){
        for (int j = 0; j < img.cols; j++){
            Vec3b px = img.at<Vec3b>(i, j);

            // Calculate the bin index for each color channel
            int binsB = std::min((int)(px[0] * bins / 256), bins - 1);
            int binsG = std::min((int)(px[1] * bins / 256), bins - 1);
            int binsR = std::min((int)(px[2] * bins / 256), bins - 1);

            // Increment the appropriate bin in the histogram
            features[binsR * bins * bins + binsG * bins + binsB] += 1;
        }
    }

    float norm = 0;
    for (int i = 0; i < features.size(); i++){
        norm += features[i];
    }

    // Normalize the histogram so that the sum of all bins equals 1
    for (int i = 0; i < features.size(); i++){
        features[i] /= norm;
    }

    return 0;
}

//Function to calculate HSV Histogram
int histHSV(Mat& image, const int bins, vector<float>& features) {
    // Convert BGR image to HSV color space
    Mat hsvImage;
    cvtColor(image, hsvImage, COLOR_BGR2HSV);

    // Calculate the bin sizes for H, S, and V channels
    // Note: H ranges from 0 to 180 in OpenCV, S and V from 0 to 255
    int hBins = bins, sBins = bins, vBins = bins;
    int histSize[] = {hBins, sBins, vBins};

    // Hue varies from 0 to 180, saturation and value from 0 to 255
    float hRanges[] = {0, 180};
    float sRanges[] = {0, 256};
    float vRanges[] = {0, 256};

    const float* ranges[] = {hRanges, sRanges, vRanges};
    int channels[] = {0, 1, 2}; // Indexes of channels to be used for the histogram

    // Create the histogram
    Mat hist;
    calcHist(&hsvImage, 1, channels, Mat(), hist, 3, histSize, ranges, true, false);

    // Normalize the histogram
    hist /= hsvImage.total(); // Normalizing by the number of pixels in the image

    // Convert the 3D histogram to a 1D feature vector
    features.clear();
    features.reserve(hBins * sBins * vBins);
    for (int h = 0; h < hBins; ++h) {
        for (int s = 0; s < sBins; ++s) {
            for (int v = 0; v < vBins; ++v) {
                int idx[] = {h, s, v};
                features.push_back(hist.at<float>(idx));
            }
        }
    }

    return 0;
}

// Helper function to shift the DFT image
void shiftDFT(Mat& src, Mat& dst) {
    dst = src.clone();
    int cx = dst.cols / 2;
    int cy = dst.rows / 2;

    Mat q0(dst, Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
    Mat q1(dst, Rect(cx, 0, cx, cy));  // Top-Right
    Mat q2(dst, Rect(0, cy, cx, cy));  // Bottom-Left
    Mat q3(dst, Rect(cx, cy, cx, cy)); // Bottom-Right

    Mat tmp; // Swap quadrants (Top-Left with Bottom-Right)
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);

    q1.copyTo(tmp); // Swap quadrant (Top-Right with Bottom-Left)
    q2.copyTo(q1);
    tmp.copyTo(q2);
}


//Function to compute features from a fourier transform of image
void fourierFeatures(Mat& inputImage, vector<float>& features) {
    // Ensure features vector is empty
    features.clear();

    // Convert to grayscale
    Mat gray;
    if (inputImage.channels() > 1) {
        cvtColor(inputImage, gray, COLOR_BGR2GRAY);
    } else {
        gray = inputImage.clone();
    }

    // Convert to float
    Mat floatImage;
    gray.convertTo(floatImage, CV_32F);

    // Apply DFT
    Mat dftImage;
    dft(floatImage, dftImage, DFT_COMPLEX_OUTPUT);

    // Shift the zero frequency component to the center
    Mat shiftedDft;
    shiftDFT(dftImage, shiftedDft);

    // Calculate magnitude
    vector<Mat> planes;
    split(shiftedDft, planes); // planes[0] = Re(DFT(I)), planes[1] = Im(DFT(I))
    magnitude(planes[0], planes[1], planes[0]);
    Mat mag = planes[0];

    // Log scale for visibility
    mag += cv::Scalar::all(1);
    log(mag, mag);

    // Normalize
    normalize(mag, mag, 0, 1, NORM_MINMAX);

    // Resize to 16x16
    Mat resizedMag;
    resize(mag, resizedMag, Size(16, 16), 0, 0, INTER_LINEAR);

    // Store in vector
    resizedMag.reshape(0, 1).copyTo(features); // Flatten and copy to output vector
}


//Function to compute features from gabor filter
void gaborFeatures(Mat& img, vector<float>& features) {
    Mat gray;
    if (img.channels() > 1) {
        cvtColor(img, gray, COLOR_BGR2GRAY);
    } else {
        gray = img.clone();
    }
    // Define Gabor filter parameters
    int kernel_size = 16;
    double sigma = 2.5;
    double lambda = 50.0;
    double gamma = 0.5;
    double theta = 0;
    double psi = 90;
    Mat dst;
    Mat kernel = cv::getGaborKernel(cv::Size(kernel_size, kernel_size), sigma, theta, lambda, gamma, psi, CV_32F);
    cv::filter2D(gray, dst, CV_32F, kernel);

    //Store mean and std dev as features
    cv::Scalar mean, stddev;
    cv::meanStdDev(dst, mean, stddev);

    features.push_back(static_cast<float>(mean[0]));
    features.push_back(static_cast<float>(stddev[0]));
}

//Function to calculate RG chromaticity histogram
int hist2d(Mat &img, const int bins, vector<float> &features){

    // initialize the histogram to 4 byte floating point, ranged 0.0-1.0, bin_size x bin_size matrix (R & G chromaticity)
    Mat hist = Mat::zeros(bins, bins, CV_32F);
    int totalRows = img.rows;
    int totalCols = img.cols;

    // calculate the RG chromaticity for each pixel and update the histogram
    for(int row = 0; row < totalRows; row++) {
        for(int col = 0; col < totalCols; col++) {

            Vec3b color = img.at<Vec3b>(row, col);

            float R = color[2];
            float G = color[1];
            float B = color[0];
            float rgbsum = R + G + B;

            // check if rgbsum is not 0, because if it is, we avoid division by 0
            if(rgbsum != 0) {

                // get the proportional red and green vals
                float r = R / rgbsum;
                float g = G / rgbsum;

                // determine the bin index for the normalized red and green chromes,
                // make sure it's within the bin range 0 - bin_size
                int rBin = min(static_cast<int>(r * bins), bins - 1);
                int gBin = min(static_cast<int>(g * bins), bins - 1);

                // add to the counter for the corresponding rBin and gBin value in the hist matrix
                hist.at<float>(rBin, gBin) += 1.0;
            }
        }
    }

    // normalize every value of the bin x bin histogram for fair comparison
    hist /= totalRows * totalCols;

    // convert histogram to a feature vector
    features.clear();
    features.reserve(bins * bins);
    for(int r = 0; r < bins; r++) {
        for(int g = 0; g < bins; g++) {
            features.push_back(hist.at<float>(r, g));
        }
    }

    return 0; // success
}

//Alternate method for 3d histogram
int colorHist(Mat &img, int bins, vector<Mat> &hist){
    // define range for color histogram calculation
    float pixelValRange[] = {0, 256};
    const float* histRange[] = { pixelValRange };
    bool uniform = true, accumulate = false;

    // create empty matrices for histograms of each color channel
    Mat histBlue, histGreen, histRed;

    // create three corresponding color channels of same size as input img but empty, make them single channel
    Mat blueChannel = Mat::zeros(img.size(), CV_8UC1);
    Mat greenChannel = Mat::zeros(img.size(), CV_8UC1);
    Mat redChannel = Mat::zeros(img.size(), CV_8UC1);

    int totalRows = img.rows;
    int totalCols = img.cols;

    // separate the color channels
    for (int row = 0; row < totalRows; row++) {
        for (int col = 0; col < totalCols; col++) {
            // get current pixel's color
            Vec3b color = img.at<Vec3b>(row, col);
            // assign each channel to its corresponding image channel
            redChannel.at<uchar>(row, col) = color[2];
            greenChannel.at<uchar>(row, col) = color[1];
            blueChannel.at<uchar>(row, col) = color[0];
        }
    }

    // calculate the histogram for each channel
    // pointer to the input img channel, num of source imgs,channel number (0) - each calculated separately,
    // mask - empty matrix since we're calculating full img, output hist, hist dimension (1D), pointer to num of bins
    // hist range of values, uniform (true), accumulate (false - hist cleared everytime in the beginning)
    calcHist(&blueChannel, 1, 0, Mat(), histBlue, 1, &bins, histRange, uniform, accumulate);
    calcHist(&greenChannel, 1, 0, Mat(), histGreen, 1, &bins, histRange, uniform, accumulate);
    calcHist(&redChannel, 1, 0, Mat(), histRed, 1, &bins, histRange, uniform, accumulate);

    // normalize the histograms so that their values sum up to 1
    normalize(histGreen, histGreen, 1.0, 0.0, NORM_L1);
    normalize(histBlue, histBlue, 1.0, 0.0, NORM_L1);
    normalize(histRed, histRed, 1.0, 0.0, NORM_L1);

    // store the calculated histograms in the vector
    hist.push_back(histBlue);
    hist.push_back(histGreen);
    hist.push_back(histRed);

    return 0; // success
}

/*
 * General function for applying the sobel filters. Since the X and Y sobel filters are transposes of each other,
 * We can interchange the horizontal and vertical kernals to get the required direction of sobel filter.
 */

int sobel(cv::Mat &src, cv::Mat &dst, const int hk[3],const int vk[3],const int norm[2]) {

    Mat inter;
    inter.create(src.size(), CV_16SC3); // allocate space for an intermediary mat
    // (IMPORTANT: To prevent heap corruption error)

    for (int i = 1; i < src.rows-1; i++){//vertical filter application
        cv::Vec3b *top = src.ptr<cv::Vec3b>(i-1); //pointers to top, mid and bottom rows
        cv::Vec3b *mid = src.ptr<cv::Vec3b>(i);
        cv::Vec3b *bot = src.ptr<cv::Vec3b>(i+1);
        cv::Vec3s *iptr = inter.ptr<cv::Vec3s>(i); //pointer to intermediary variable row

        for(int j = 0; j < src.cols; j++){//cols
            for (int k = 0; k < src.channels(); k++){//color channels
                iptr[j][k] = vk[0] * bot[j][k] + vk[1] * mid[j][k] + vk[2] * top[j][k];//make sure to normalize
            }
        }

    }

    dst.create(inter.size(), CV_16SC3);// allocate space for dst
    for (int i = 0; i < inter.rows; i++) {// Apply horizontal filter
        cv::Vec3s *ptr = inter.ptr<cv::Vec3s>(i); //source pointer (intermediary)
        cv::Vec3s *dptr = dst.ptr<cv::Vec3s>(i); //dst pointer
        for (int j = 1; j < inter.cols-1; j++){
            for(int k = 0; k < inter.channels(); k++){
                dptr[j][k] = hk[0] * ptr[j-1][k] + hk[1] * ptr[j][k] + hk[2] * ptr[j+1][k];//make sure to normalize
            }
        }

    }
    return 0;
}

int sobelX3x3 (cv::Mat &src, cv::Mat &dst){
    int hk[3] = {-1, 0, 1};
    int vk[3] = {1, 2, 1};
    const int norm[2]={1,4};
    sobel(src,dst,hk,vk,norm);
    return 0;
}

/*
 * Sobel Y filter
 *          [ 1;   [1  2  1;
 *[1 2 1] x   0; =  0  0  0;
 *           -1]   -1 -2 -1]
 */


int sobelY3x3 (cv::Mat &src, cv::Mat &dst){
    int vk[3] = {1, 0, -1};
    int hk[3] = {1, 2, 1};
    const int norm[2]={4,1}; //h,v
    sobel(src,dst,hk,vk,norm);
    return 0;
}

/*
 * Magnitude image is generated by taking magnitude of each pixel
 * of sobel gradients in x and y direction
 */
int calcMagnitude( cv::Mat &sx, cv::Mat &sy, cv::Mat &dst ){
    dst.create(sx.size(), CV_16SC3);
    for (int i=0;i < sx.rows; i++){
        cv::Vec3s *xptr = sx.ptr<cv::Vec3s>(i);//to acces sx rows
        cv::Vec3s *yptr = sy.ptr<cv::Vec3s>(i);//to acces sy rows
        cv::Vec3s *dptr = dst.ptr<cv::Vec3s>(i);//to acces dst rows
        for(int j=0; j<sx.cols; j++){
            for(int k=0; k<sx.channels(); k++){
                dptr[j][k]=sqrt((xptr[j][k]*xptr[j][k])+(yptr[j][k]*yptr[j][k]));
            }
        }

    }
    dst.convertTo(dst, CV_8U); // convert to 8-bit unsigned integer
    return 0;
}


int textureHist(Mat &img, int bins, vector<Mat> &hist) {
    Mat gray, sobelX, sobelY, magnitude;
    bool uniform = true, accumulate = false;
    float pixelValRange[] = {0, 256};
    const float* histRange[] = { pixelValRange };
    int histSize[] = { bins };
    Mat histTemp;

    // apply custom Sobel filters to calculate gradients in x and y directions
    sobelX3x3(img, sobelX);
    sobelY3x3(img, sobelY);
    // use gradients x & y to get the magnitude
    calcMagnitude(sobelX, sobelY, magnitude);

    //Normalize
    int channels[] = {0};
    calcHist(&magnitude, 1, channels, Mat(), histTemp, 1, histSize, histRange, uniform, accumulate);
    normalize(histTemp, histTemp, 1.0, 0.0, NORM_L1); // Normalize to make the sum of bins equal 1

    hist.clear();
    hist.push_back(histTemp); // histogram stored back in the expected vector<Mat> format
    return 0; // success
}

// function to flatten and merge histograms from vector<Mat> to vector<float>
vector<float> mergeHistograms(const vector<Mat> &colorHist, const vector<Mat> &textureHist) {
    vector<float> features;

    // flatten color histograms (R, G, B) and append to the feature vector
    for (const Mat &h : colorHist) {
        for (int i = 0; i < h.rows; i++) {
            features.push_back(h.at<float>(i));
        }
    }

    // flatten texture histogram and append to the feature vector
    for (const Mat &h : textureHist) {
        for (int i = 0; i < h.rows; i++) {
            features.push_back(h.at<float>(i));
        }
    }

    return features;
}

// function to extract and combine color and texture histograms into a feature vector
int colorTextureHist(Mat &img, int bins, vector<float> &features) {
    vector<Mat> colorHists, textureHists;

    // extract histograms
    colorHist(img, bins, colorHists);
    textureHist(img, bins, textureHists);

    // merge and flatten histograms into a single feature vector
    features = mergeHistograms(colorHists, textureHists);

    return 0; // success
}
