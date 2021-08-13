#pragma once
#include <iostream>
#include <fstream>
#include <ios>
#include "Network.h"
#include "Layers.h"
#include "node.h"
#include "Activations.h"

using namespace std;
using namespace ntwrk;

void LoadModel(string fileName, Network network) {
    std::ifstream file;

    // Opening file in input mode
    file.open(fileName, std::ios::in);

    // Reading from file into object "obj"
    file.read((char*)&network, sizeof(network));
}

void GetData(string fileName, vector<vector<float>>* images, vector<vector<float>>* labels) {
    ifstream fin;
    fin.open(fileName);
    vector<vector<float>> all;
    vector<float> singleIm;
    vector<float> _labels;
    string line;
    string temp = "";
    int t, label;
    while (!fin.eof()) {
        label = 0;
        line = "";
        getline(fin, line);
        if (line == "") {
            break;
        }

        for (char& c : line) {
            if (c != ',') {
                temp += c;
            }
            else {
                float pixel = std::stof(temp);
                if (label == 0) {
                    label = 1;
                }
                else {
                    pixel /= 255;
                }
                singleIm.push_back(pixel);
                temp = "";
            }
        }
        t = singleIm[0];
        singleIm.erase(singleIm.begin());
        _labels.clear();
        for (int i = 0; i < 10; i++) {
            if (i == t) {
                _labels.push_back(1);
            }
            else {
                _labels.push_back(0);
            }
        }
        (*labels).push_back(_labels);
        (*images).push_back(singleIm);
        singleIm.clear();
    }
}

int main() {
    vector<int> tSeriesDat = { 0, 13, 9, 4, 8, 12, 13, 31, 35, 8, 73, 77, 21, 104, 105, 92, 192, 2, 67, 283, 135, 129, 203, 152, 107, 185, 110, 138, 100, 228, 130, 137, 235, 138, 166, 122, 149, 124, 187, 195, 130, 126, 154, 149, 106, 151 ,100, 91, 46, 38, 13, 22, 33, 9, 14, 48, 7, 18, 6, 21, 30, 21, 79, 64, 55, 18, 101, 63, 59, 6, 26, 150, 80, 82, 58, 84, 59, 60, 116, 99, 112, 197, 116, 215, 82, 155, 80, 13, 70, 391, 178, 124, 235, 88, 187, 140, 111, 72, 75, 44, 40, 37, 10, 6, 7, 5, 10, 11, 16, 20, 17, 21, 34, 28, 44, 33, 85, 16, 95, 43, 35, 30, 259 };

    Network myNetwork;

   /* Input inp(783);
    myNetwork.SetInput(&inp);
    //Dense layer1(300, { 0 }, "sigmoid");
    //Dense layer2(100, { 1 }, "sigmoid");
    //Dense layer3(50, { 2 }, "sigmoid");
    //Dense layer4(25, { 3 }, "sigmoid");
    //Dense layer5(10, { 4 }, "softmax");
    myNetwork.AddLayer(new Dense(300, { 0 }, "sigmoid"));//&layer1);//input layers start from 1
    myNetwork.AddLayer(new Dense(100, { 1 }, "sigmoid"));
    myNetwork.AddLayer(new Dense(50, { 2 }, "sigmoid"));
    myNetwork.AddLayer(new Dense(25, { 3 }, "sigmoid"));
    myNetwork.AddLayer(new Dense(10, { 4 }, "softmax"));
    */

    myNetwork.SetInput(783);
    myNetwork.AddLayer(300, { 0 }, "sigmoid");//&layer1);//input layers start from 1
    myNetwork.AddLayer(100, { 1 }, "sigmoid");
    myNetwork.AddLayer(50, { 2 }, "sigmoid");
    myNetwork.AddLayer(50, { 3 }, "sigmoid");
    myNetwork.AddLayer(50, { 4 }, "sigmoid");
    myNetwork.AddLayer(25, { 5 }, "sigmoid");
    myNetwork.AddLayer(10, { 6 }, "softmax");

    myNetwork.Compile("CatCrossEntro");

    vector<vector<float>> Trainimages;
    vector<vector<float>> Trainlabels;
    vector<vector<float>> Testimages;
    vector<vector<float>> Testlabels;

    // for PC
    GetData("C:/Users/Neo/Documents/MnistFashion/fashion-mnist_test.csv", &Testimages, &Testlabels);
    GetData("C:/Users/Neo/Documents/MnistFashion/fashion-mnist_train.csv", &Trainimages, &Trainlabels);

    // for Laptop
    //GetData("C:/Users/nsmne/Documents/MnistFashion/fashion-mnist_test.csv", &Testimages, &Testlabels);
    //GetData("C:/Users/nsmne/Documents/MnistFashion/fashion-mnist_train.csv", &Trainimages, &Trainlabels);

    myNetwork.Train(Trainimages, Trainlabels, 1, 8);

    myNetwork.Test(Testimages, Testlabels);

    //myNetwork.SaveModel("myModel");

    //myNetwork.LoadModel("myModel");
    //cout << "test" << endl;
}