/*
 * CS 5330: Pattern Recognition And Computer Vision
 * Spring 2024
 * Project 2: Content Based Image Recognition
 * Authors: Adnan Amir & Tenzin Kunsang
 * Purpose of file: Main file which implements the cbir pipeline
 */

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include "features.h"
#include "im2csv.h"
#include "metrics.h"
#include "csv_util.h"
#include <dirent.h>
#include <vector>
#include <iostream>
#include <algorithm>

using namespace cv;
using namespace std;

//copied from https://stackoverflow.com/questions/2896600/how-to-replace-all-occurrences-of-a-character-in-string
std::string ReplaceAll(std::string str, const std::string& from, const std::string& to) {
    size_t start_pos = 0;
    while ((start_pos = str.find(from, start_pos)) != std::string::npos) {
        str.replace(start_pos, from.length(), to);
        start_pos += to.length();
    }
    return str;
}


/*
 * The entire process can be implemented as a command line program
 * that takes in a target filename for T,
 * a directory of images as the database B,
 * the feature type, 1-bline, 2-schist, 3-mchist, 4-txhist, 5-DN, 7-Custom
 * the matching method, 1-ssd, 2-histint, 3-cosdist
 * the number of images N to return.
 */

int cbir_pipeline(const string &T, //target filename
                  const string &B, //target database
                  const int featuretype,
                  const int distmetric,
                  const int top_n,
                  const int bot_n){
    Mat img = imread(T,IMREAD_COLOR);
    if (img.empty()){
        cerr << "Image not read" << T << endl;
    }

    vector <float> target_features;
    int f1,f2;

    switch (featuretype) {
        case 1: {
            classic(img,target_features);
            break;
        }
        case 2: {
            int bins = 16;
            hist2d(img, bins, target_features);
            break;
        }
        case 3:{
            vector <float> target_features2;
            cv::Rect top_half(0,0,img.cols,img.rows/2);
            cv::Rect bot_half(0,img.rows/2,img.cols,img.rows/2);
            cv::Mat th_img=img(top_half);
            cv::Mat bh_img=img(bot_half);
            histogram3dfeatures(th_img,8,target_features);//top
            histogram3dfeatures(bh_img,8,target_features2);//bot
            target_features.insert(target_features.end(),target_features2.begin(),target_features2.end());
            break;
        }
        case 4: {
            int bins = 256;
            colorTextureHist(img, bins, target_features);
            break;
        }
        case 5:{
            vector<vector<float>> temp;
            vector<char*> file;
            read_image_data_csv(const_cast<char*>(B.c_str()), file, temp);
            for (size_t i=0; i<temp.size(); i++) {
                string filei = "E:/backup/Desktop/College/NEU/sem 2/cs 5330 pattern recognition and computer vision/Project2/olympus/" + string(file[i]);
                if (T == filei){
                    target_features = temp[i];
                    cout << "Features Written" << endl;
                }
            }
            break;
        }
        case 7:{
            Mat gray;
            vector<float> feat1,feat2,feat3;
            //get grayscale image
            cvtColor(img,gray,COLOR_BGR2GRAY);

            //get fourier features
            fourierFeatures(gray,feat1);
            f1 = feat1.size();

            //get gabor features
            gaborFeatures(gray,feat2);
            f2 = feat2.size();

            //get HSV histogram
            histHSV(img,8,feat3);

//            appending to target features
            target_features = feat1;
            target_features.insert(target_features.end(),feat2.begin(),feat2.end());
            target_features.insert(target_features.end(),feat3.begin(),feat3.end());
            feat1.clear();
            feat2.clear();
            feat3.clear();
            break;
        }
    }
    vector<char*> dbn; //database file name
    vector<vector<float>> dbf; //database file features

    read_image_data_csv(const_cast<char*>(B.c_str()), dbn, dbf);
    vector<pair<float, string>> results;


    for (size_t i=0; i<dbf.size(); i++){
        string dbi = ReplaceAll(string(dbn[i]),"\\","/");
        if (featuretype == 5){
            dbi = "E:/backup/Desktop/College/NEU/sem 2/cs 5330 pattern recognition and computer vision/Project2/olympus/" + string(dbn[i]);
        }
        if (T != dbi) {
            switch (distmetric) {
                case 1:{ //Euclidean Distance
                    float dis = ssd(target_features,dbf[i]);
                    if (dis != -1){
                        results.push_back({dis,dbi});
                    }
                    break;
                }
                case 2:{ // Histogram intersection
                    float inter = histogram_intersection(target_features,dbf[i]);
                    results.push_back({inter,dbi});
                    break;
                }
                case 3:{//mchist
                    int l= target_features.size()/2;
                    vector<float> tbot,ttop,cbot,ctop;
                    for (int k=0;k<l;k++){
                        ttop.push_back(target_features[k]);
                        tbot.push_back(target_features[l+k]);
                        ctop.push_back(dbf[i][k]);
                        cbot.push_back(dbf[i][l+k]);
                    }
                    float top_inter = histogram_intersection(ttop,ctop);
                    float bot_inter = histogram_intersection(tbot,cbot);
                    float combined_inter = 0.5 * top_inter + 0.5 * bot_inter;
                    results.push_back({combined_inter,dbi});
                    tbot.clear();
                    cbot.clear();
                    ttop.clear();
                    ctop.clear();
                    break;
                }
                case 4: {
                    float dis = textureColorHistDistance(target_features, dbf[i]);
                    if (dis != -1){
                        results.push_back({dis,dbi});
                    }
                    break;
                }
                case 5:{ // Cosine distance
                    float dis = cosdist(target_features,dbf[i]);
                    results.push_back({dis,dbi});
                    break;
                }
                case 7:{
                    float df1,df2,df3;
                    vector<float> tf1,tf2,tf3,cf1,cf2,cf3;
                    for (int k =0; k<f1;k++){
                        tf1.push_back(target_features[k]);
                        cf1.push_back(dbf[i][k]);
                    }
                    for (int k =f1; k<(f1+f2);k++){
                        tf2.push_back(target_features[k]);
                        cf2.push_back(dbf[i][k]);
                    }
                    for (int k =f2; k<target_features.size();k++){
                        tf3.push_back(target_features[k]);
                        cf3.push_back(dbf[i][k]);
                    }
                    df1 = cosdist(tf1,cf1);
                    df2 = cosdist(tf2,cf2);
                    df3 = chiSquaredDistance(tf3,cf3);

                    float combined = 0.3*df1+0.2*df2+0.5*df3;
                    results.push_back({combined,dbi});
                    break;
                }

            }
        }

    }

    if (distmetric == 1 || distmetric == 5 || distmetric == 7){//sort L2 or cosine distances in ascending order
        sort(results.begin(),results.end());
    }
    if (distmetric == 2 || distmetric == 3 || distmetric == 4){// sort histogram based intersections in decreasing order
        sort(results.begin(),results.end(),[](const auto & a,const auto & b){
            return a.first > b.first;
        });
    }

    imshow("Target",img);
    cout << "Top " << top_n << " images similar to target image are:" << endl;
    for (int i = 0; i<min(top_n, (int)results.size()); i++){
        string path = results[i].second;
        Mat dst = imread(path,IMREAD_COLOR);
        string win = "Image " + to_string(i+1);
        imshow(win, dst);
        cout << path << endl;
        waitKey(0);
    }
    cout << "Top " << bot_n << " images least similar to target image are:" << endl;
    for (int i = results.size()-1; i>results.size()-bot_n-1; i--){
        string path = results[i].second;
        Mat dst = imread(path,IMREAD_COLOR);
        string win = "Image " + to_string(i+1);
        imshow(win, dst);
        cout << path << endl;
        waitKey(0);
    }

    results.clear();
    return 0;
}

/*
 * The entire process can be implemented as a command line program
 * that takes in a target filename for T,
 * a directory of images as the database B,
 * the feature type, 1-bline, 2-schist, 3-mchist, 4-txhist, 5-DN, 7-Custom
 * the matching method, 1-ssd, 2-histint,3-mchist weighted dist , 4-texturecolor hist, 5-cosine dist, 7-Custom weighted dist;
 * the number of images N to return.
 */

int main() {
//    imdb2csv(DB_PATH,CUSTOM_CSV_PATH,7);
    string T = "E:/backup/Desktop/College/NEU/sem 2/cs 5330 pattern recognition and computer vision/Project2/olympus/pic.0343.jpg";
    string B = "E:/backup/Desktop/College/NEU/sem 2/cs 5330 pattern recognition and computer vision/Project2/database/classic.csv";
    int featuretype = 1;
    int distmetric = 1;
    int top_n = 5;
    int bot_n = 5;

    cbir_pipeline(T,B,featuretype,distmetric,top_n,bot_n);

    return 0;
//    int main(int argc, char *argv[])
//    // check if sufficient arguments were provided
//    if (argc < 2) {
//        cout << "Usage: " << argv[0] << " <featuretype> " << endl;
//        return -1;
//    }
//
//    int featuretype = atoi(argv[1]); // parse command line argument and assign to ft
//    int distmetric = featuretype;
//    int top_n = 3; // default top 3
//    int bot_n = 3; //default last 3
//
//    string basePath = "/Users/kunsang/Desktop/1computervision/assignments/TenzinKunsangProject2/";
//    string T, B;
//
//    switch (featuretype) {
//        case 1:
//            cout << "Feature type 1 selected." << endl;
//            T = basePath + "olympus/pic.1016.jpg";
//            B = basePath + "database/classic.csv";
//            break;
//        case 2:
//            cout << "Feature type 2 selected." << endl;
//            T = basePath + "olympus/pic.0164.jpg";
//            B = basePath + "database/hist2d.csv";
//            break;
//        case 3:
//            cout << "Feature type 3 selected." << endl;\
//            T = basePath + "olympus/pic.0274.jpg";
//            B = basePath + "database/multi.csv";
//            break;
//        case 4:
//            cout << "Feature type 4 selected." << endl;
//            T = basePath + "olympus/pic.0535.jpg";
//            B = basePath + "database/colorTextureHist.csv";
//            break;
//        case 5:
//            cout << "Feature type 5 selected." << endl;
//            T = basePath + "olympus/pic.0164.jpg"; // images - 0893 and 0164
//            B = basePath + "database/ResNet18_olym.csv";
//            break;
//        case 7:
//            cout << "Feature type 7 selected." << endl;
//            T = basePath + "olympus/pic.0343.jpg"; // images - 0893 and 0164
//            B = basePath + "database/custom.csv";
//            break;
//        default:
//            cout << "Invalid feature type entered. Exiting." << endl;
//            return -1;
//    }
//
//    if (!T.empty() && !B.empty()) {
//        cbir_pipeline(T, B, featuretype, distmetric, top_n, bot_n);
//    }

}

