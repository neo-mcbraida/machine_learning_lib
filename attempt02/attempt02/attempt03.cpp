#pragma once
#include <iostream>
#include <fstream>
#include <ios>
#include "Network.h"
#include "Layers.h"
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
    Network myNetwork;
    Input inp(783);
    myNetwork.Input(&inp);
    Dense layer1(300, "relu");
    Dense layer2(100, "relu");
    Dense layer3(50, "relu");
    Dense layer4(25, "relu");
    myNetwork.AddHiddenLayer(&layer1);//input layers start from 1
    Dense layer5(10, "relu");
    myNetwork.AddHiddenLayer(&layer2);
    myNetwork.AddHiddenLayer(&layer3);
    myNetwork.AddHiddenLayer(&layer4);
    myNetwork.AddHiddenLayer(&layer5);
    //vector<float> inputData = { 1, 1 };

    vector<vector<float>> Trainimages;
    vector<vector<float>> Trainlabels;
    vector<vector<float>> Testimages;
    vector<vector<float>> Testlabels;
    GetData("C:/Users/nsmne/Documents/machine_learning_lib/fashion-mnist_test.csv", &Testimages, &Testlabels);
    GetData("C:/Users/nsmne/Documents/machine_learning_lib/fashion-mnist_train.csv", &Trainimages, &Trainlabels);
}