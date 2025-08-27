/*
 * CS 5330: Pattern Recognition And Computer Vision
 * Spring 2024
 * Project 2: Content Based Image Recognition
 * Authors: Adnan Amir & Tenzin Kunsang
 * Purpose of file: Implements Function to convert database images into CSV format
 * Please configure definitions in im2csv.h
 */

#include "../include/im2csv.h"
#include "../include/features.h"
#include "../include/csv_util.h"
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <vector>
#include <algorithm>
using namespace std;
using namespace std::filesystem;

int imdb2csv(string db, char filename[], int featuretype) {

    enum ft {
        CLASSIC = 1,
        HG2D = 2,
        MULTI = 3,
        CTHG = 4,
        CUSTOM = 7,
    } ft;

    // Delete the existing file to avoid appending to old data
    if (exists(filename)) {
        remove(filename); // Delete the file if it exists
    }

    for(const auto& line : directory_iterator(db)){
        string img_path = line.path().string();
        string ext = line.path().extension().string();

        //Check if file is an image
        if (ext != ".jpg" && ext != ".jpeg" && ext != ".png" && ext != ".tiff"){
            cerr << "Non image file -> Skipping:" << img_path << endl;
            continue;
        }

        //Read Image
        cv::Mat img = cv::imread(img_path, cv::IMREAD_COLOR);
        if (img.empty()) {
            cerr << "Could not read the image: " << img_path << endl;
            continue;
        }
        //Initialize Feature Vector
        vector<float> features;

        switch (featuretype) {
            case 1:ft = CLASSIC;
                break;
            case 2:ft = HG2D;
                break;
            case 3:ft = MULTI;
                break;
            case 4:ft = CTHG;
                break;
            case 7:ft = CUSTOM;
                break;
        }

        switch (ft) {
            case CLASSIC: {
                classic(img,features); //compute features
                break;
            }
            case HG2D:{
                hist2d(img,16,features);
                break;
            }
            case MULTI:{
                cv::Rect top_half(0,0,img.cols,img.rows/2);
                cv::Rect bot_half(0,img.rows/2,img.cols,img.rows/2);
                cv::Mat th_img=img(top_half);
                cv::Mat bh_img=img(bot_half);
                vector<float> top_features,bot_features;
                histogram3dfeatures(th_img,8,top_features);
                histogram3dfeatures(bh_img,8,bot_features);
                features = top_features;
                features.insert(features.end(),bot_features.begin(),bot_features.end());
                top_features.clear();
                bot_features.clear();
                break;
            }
            case CTHG:{
                colorTextureHist(img, 256, features);
                break;
            }
            case CUSTOM:{
                cv::Mat gray;
                vector<float> feat1,feat2,feat3;
//                int center_x = img.cols / 2;
//                int center_y = img.rows / 2;
//                cv::Rect roi(0, 0, img.cols, img.rows);
//                cv::Mat feature_mat = img(roi).clone();
                //get grayscale image
                cv::cvtColor(img,gray,cv::COLOR_BGR2GRAY);

                //get fourier features
                fourierFeatures(gray,feat1);

                //get gabor features
                gaborFeatures(gray,feat2);

                //get HSV histogram
                histHSV(img,8,feat3);

//            appending to features
                features = feat1;
                features.insert(features.end(),feat2.begin(),feat2.end());
                features.insert(features.end(),feat3.begin(),feat3.end());
                feat1.clear();
                feat2.clear();
                feat3.clear();
                break;
            }

        }
        // Write the extracted features and image path to the CSV file
        append_image_data_csv(filename, const_cast<char*>(img_path.c_str()), features);
        features.clear(); // Clear the feature vector for the next image
    }

    return 0;
}