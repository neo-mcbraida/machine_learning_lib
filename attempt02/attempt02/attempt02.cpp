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
    Network myNetwork;
    Input inp(783);
    myNetwork.SetInput(&inp);
    Dense layer1(300, { 0 }, "sigmoid");
    Dense layer2(100, { 1 }, "sigmoid");
    Dense layer3(50, { 2 }, "sigmoid");
    Dense layer4(25, { 3 }, "sigmoid");
    Dense layer5(10, { 4 }, "softmax");
    myNetwork.AddLayer(&layer1);//input layers start from 1
    myNetwork.AddLayer(&layer2);
    myNetwork.AddLayer(&layer3);
    myNetwork.AddLayer(&layer4);
    myNetwork.AddLayer(&layer5);
    //vector<float> inputData = { 1, 1 };

    myNetwork.Compile(new CatCrossEntro);

    vector<vector<float>> Trainimages;
    vector<vector<float>> Trainlabels;
    vector<vector<float>> Testimages;
    vector<vector<float>> Testlabels;

    GetData("C:/Users/Neo/Documents/MnistFashion/fashion-mnist_test.csv", &Testimages, &Testlabels);
    GetData("C:/Users/Neo/Documents/MnistFashion/fashion-mnist_train.csv", &Trainimages, &Trainlabels);

    //GetData("C:/Users/nsmne/Documents/MnistFashion/fashion-mnist_test.csv", &Testimages, &Testlabels);
    //GetData("C:/Users/nsmne/Documents/MnistFashion/fashion-mnist_train.csv", &Trainimages, &Trainlabels);

    myNetwork.Train(Trainimages, Trainlabels, 3, 8);
}