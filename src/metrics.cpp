/*
 * CS 5330: Pattern Recognition And Computer Vision
 * Spring 2024
 * Project 2: Content Based Image Recognition
 * Authors: Adnan Amir & Tenzin Kunsang
 * Purpose of file: File containing functions for matching different feature vectors
 */

#include "../include/metrics.h"
#include <vector>
#include <iostream>
#include <cmath>
#include <numeric>
#include <stdexcept>

using namespace std;
float ssd(vector<float> &target, vector<float> &dbi){
    if (target.size() != dbi.size()){
        cerr << "Feature Vectors not same size" << endl;
        return -1;
    }

    float dist = 0;
    for (size_t i = 0; i < target.size(); i++){
        float dx = target[i] - dbi [i];
        dist += pow(dx,2);
    }
    return dist;
}

float histogram_intersection(vector<float> &target, vector<float> &dbi){
    float inter=0;
    for (size_t i=0; i < target.size(); i++){
        inter += min(target[i],dbi[i]);
    }
    return inter;
}

float cosdist(vector<float> &target, vector<float> &dbi){
    if (target.size() != dbi.size()){
        throw invalid_argument("Feature Vectors not same size");
    }

    float dot= inner_product(target.begin(),target.end(),dbi.begin(),0.0f);
    float mag_target = sqrt(inner_product(target.begin(),target.end(),target.begin(),0.0f));
    float mag_dbi = sqrt(inner_product(dbi.begin(),dbi.end(),dbi.begin(),0.0f));

    if (mag_dbi == 0 || mag_target == 0) {
        throw invalid_argument("One of the vector has 0 magnitude so cosine distance cannot be computed");
    }

    float similar=dot/(mag_dbi*mag_target); //cosine similarity
    return 1-similar;// cosine distance
}

// Function to compute Chi-squared distance between two histograms
double chiSquaredDistance(const vector<float>& H1, const vector<float>& H2) {
    if (H1.size() != H2.size()) {
        cerr << "Error: Histograms do not have the same size." << endl;
        return -1; // Return an error code or handle error appropriately
    }

    double distance = 0.0;
    for (size_t i = 0; i < H1.size(); ++i) {
        if (H1[i] + H2[i] == 0) continue; // Avoid division by zero
        distance += (pow(H1[i] - H2[i], 2) / (H1[i] + H2[i]));
    }

    // Normalize the distance to [0, 1] range
    double normalizedDistance = distance / (distance + 1);
    return normalizedDistance;
}
// for similarity comparison of histogram - https://safjan.com/metrics-to-compare-histograms/
// calculate the Bhattacharyya distance between two histograms for similarity
double bhattacharyyaDistanceSegment(const vector<float> &histA, const vector<float> &histB, int start, int end) {
    double hA_sum = std::accumulate(histA.begin() + start, histA.begin() + end, 0.0);
    double hB_sum = std::accumulate(histB.begin() + start, histB.begin() + end, 0.0);
    double score = 0.0;

    for (int index = start; index < end; index++) {
        double valA = histA[index] / hA_sum;
        double valB = histB[index] / hB_sum;
        score += sqrt(valA * valB);
    }

    return sqrt(1.0 - score);
}

// function to compare two histograms by channels
double textureColorHistDistance(const vector<float> &histA, const vector<float> &histB) {
    if(histA.size() != histB.size() || histA.size() % 256 != 0) {
        throw invalid_argument("Histograms must have the same size and be a multiple of 256.");
    }

    vector<double> channelDistances;
    double totalDist = 0.0;

    // 3 segments for RGB and 4th for texture
    for (int channel = 0; channel < 4; channel++) {
        int start = channel * 256;
        int end = start + 256;

        double distance = bhattacharyyaDistanceSegment(histA, histB, start, end);
        channelDistances.push_back(distance);
    }

    for (int histIndex = 0; histIndex < channelDistances.size(); histIndex++){
        totalDist += channelDistances[histIndex];
    }

    totalDist /= 4;

    return totalDist;
}