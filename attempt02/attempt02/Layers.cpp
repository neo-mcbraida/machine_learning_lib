#pragma once
#include <iostream>
#include <vector>
#include "Layers.h"

using namespace std;
using namespace ntwrk;

Layer::Layer(int _width, string _activation) {
    prevLayer = NULL;
    nextLayer = NULL;
	width = _width;
	SetActivation(_activation);
}

void Layer::SetActivation(string _activation) {
    if (_activation == "relu") {
        Relu a;
        activation = &a;
    }
    else if (_activation == "sigmoid") {
        Sigmoid a;
        activation = &a;
    }
    else if (_activation == "softmax") {
        Softmax a;
        activation = &a;
    }
    else {
        Constant a;
        activation = &a;
    }
}

void Layer::AddNodes(vector<int> = {}) {
    for (int i = 0; i < width; i++) {
        Node node({});
        nodes.push_back(node);
    }
}

vector<Node*> Layer::GetInpNodes(vector<int> inputInds) {//// bad change this
    vector<Node*> nodes;
    if (prevLayer != NULL) {
        if (std::find(inputInds.begin(), inputInds.end(), prevLayer->index) != inputInds.end()) {
            for (Node node : prevLayer->nodes) {
                nodes.push_back(&node);
            }
        }
    }
    if (prevLayer->prevLayer != NULL) {
        vector<Node*> temp = prevLayer->prevLayer->GetInpNodes(inputInds);
        nodes.insert(nodes.end(), nodes.begin(), nodes.end());
    }
    return nodes;
}

void Layer::AddLayer(Layer* newLayer){
    if (nextLayer == NULL) {
        nextLayer = newLayer;
        (*newLayer).prevLayer = this;
    }
    else { (*nextLayer).AddLayer(newLayer); }

}

Dense::Dense(int width, vector<int> _inputIndexes, string activation) : Layer(width, activation){
    inpIndexes = _inputIndexes;
}

void Dense::ForwardProp() {
    activation->SetNodeActivation(nodes);
    if (nextLayer != NULL) {
        nextLayer->ForwardProp();
    }
}

void Dense::AddNodes() {
    vector<Node*> inpNodes = GetInpNodes(inpIndexes);
    std::reverse(inpNodes.begin(), inpNodes.end());
    int i = 0;
    while (i < width) {
        Node node(inpNodes);
        nodes.push_back(node);
    }
}

void Dense::StartBackProp(vector<float> desiredOut) {
    float derivActiv;
    for (int i = 0; i < nodes.size(); i++) {
        nodes[i].desiredVals.clear();
        nodes[i].desiredVals.push_back(desiredOut[i]);
        derivActiv = activation->DerivActivation(nodes[i].activation);
        nodes[i].SetPassChanges(derivActiv);
    }
    if (prevLayer->prevLayer != NULL) {
        prevLayer->BackProp();
    }
}

void Dense::BackProp() {
    float derivActiv;
    for (int i = 0; i < nodes.size(); i++) {
        derivActiv = activation->DerivActivation(nodes[i].activation);
        nodes[i].SetPassChanges(derivActiv);
    }
    if (prevLayer->prevLayer != NULL) {
        prevLayer->BackProp();
    }
}

float Dense::GetCost(vector<float> desiredOut) {
    float cost = 0;
    for (int i = 0; i < nodes.size(); i++) {
        float temp;
        temp = nodes[i].activation * desiredOut[i];
        temp = temp * temp;
        cost += temp;
    }
    return cost;
}

void Dense::SetChanges(int batchSize) {
    for (Node node : nodes) {
        node.AdjustWB(batchSize);
    }
    if (prevLayer->prevLayer != NULL) {
        prevLayer -> SetChanges(batchSize);
    }
}

Input::Input(int width) : Layer(width, "") {}

void Input::StartForwardProp(vector<float> input) {
    SetNodes(input);
    nextLayer->ForwardProp();
}

void Input::SetNodes(vector<float> input) {
    for (int i = 0; i < nodes.size(); i++) {
        nodes[i].activation = input[i];
    }
    nextLayer->ForwardProp();
}