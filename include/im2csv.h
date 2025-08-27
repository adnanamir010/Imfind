/*
 * CS 5330: Pattern Recognition And Computer Vision
 * Spring 2024
 * Project 2: Content Based Image Recognition
 * Authors: Adnan Amir & Tenzin Kunsang
 * Purpose of file: Header file for im2csv.cpp
 * Please configure paths here
 */
#include <string>

#ifndef PROJECT2_IM2CSV_H
#define PROJECT2_IM2CSV_H
#define DB_PATH "E:/backup/Desktop/College/NEU/sem 2/cs 5330 pattern recognition and computer vision/Project2/olympus"
#define CLASSIC_CSV_PATH "E:/backup/Desktop/College/NEU/sem 2/cs 5330 pattern recognition and computer vision/Project2/database/classic.csv"
#define MULTI_CSV_PATH "E:/backup/Desktop/College/NEU/sem 2/cs 5330 pattern recognition and computer vision/Project2/database/multi.csv"
#define CUSTOM_CSV_PATH "E:/backup/Desktop/College/NEU/sem 2/cs 5330 pattern recognition and computer vision/Project2/database/custom.csv"
#define HIST2D_CSV_PATH "E:/backup/Desktop/College/NEU/sem 2/cs 5330 pattern recognition and computer vision/Project2/database/hist2d.csv"


int imdb2csv(std::string db = DB_PATH, char filename[] = CLASSIC_CSV_PATH, int featuretype=1);

#endif //PROJECT2_IM2CSV_H
