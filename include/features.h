/*
 * CS 5330: Pattern Recognition And Computer Vision
 * Spring 2024
 * Project 2: Content Based Image Recognition
 * Authors: Adnan Amir & Tenzin Kunsang
 * Purpose of file: Header file for features.cpp
 */

#include "opencv2/opencv.hpp"

#ifndef PROJECT2_FEATURES_H
#define PROJECT2_FEATURES_H

int classic(cv::Mat &img, std::vector<float> &features);
int histogram3dfeatures(cv::Mat &img, const int bins, std::vector<float> &features);
void gaborFeatures(cv::Mat& img, std::vector<float>& features);
void fourierFeatures(cv::Mat& inputImage, std::vector<float>& features);
int histHSV(cv::Mat& image, const int bins, std::vector<float>& features);
int hist2d(cv::Mat &img, const int bins, std::vector<float> &features);
int colorTextureHist(cv::Mat &img, const int bins, std::vector<float> &features);



#endif //PROJECT2_FEATURES_H
