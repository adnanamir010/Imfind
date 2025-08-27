/*
 * CS 5330: Pattern Recognition And Computer Vision
 * Spring 2024
 * Project 2: Content Based Image Recognition
 * Authors: Adnan Amir & Tenzin Kunsang
 * Purpose of file: Header file for metrics.cpp
 */

#include <vector>
using namespace std;

#ifndef PROJECT2_METRICS_H
#define PROJECT2_METRICS_H

float ssd(vector<float> &target, vector<float> &dbi);
float histogram_intersection(vector<float> &target, vector<float> &dbi);
float cosdist(vector<float> &target, vector<float> &dbi);
double chiSquaredDistance(const vector<float>& H1, const vector<float>& H2);
double textureColorHistDistance(const vector<float> &h1, const vector<float> &h2); //bhattacharyyaa

#endif //PROJECT2_METRICS_H
