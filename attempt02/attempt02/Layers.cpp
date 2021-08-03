#pragma once
#include <iostream>
#include <vector>
#include "Layers.h"
#include "node.h"
#include "Activations.h"

using namespace std;
using namespace ntwrk;

Layer::Layer(int _width, string _activation) {
    //prevLayer = nullptr;
    //nextLayer = nullptr;
	width = _width;
	SetActivation(_activation);
}


void Layer::BackProp() {}
void Layer::SetChanges(int batchSize) {}
void Layer::ForwardProp() {}

void Layer::SetActivation(string _activation) {
    if (_activation == "relu") {
        Relu* a = new Relu;
        activation = a;
    }
    else if (_activation == "sigmoid") {
        Sigmoid* a = new Sigmoid;
        activation = a;
    }
    else if (_activation == "softmax") {
        Softmax* a = new Softmax;
        activation = a;
    }
    else {
        Constant* a = new Constant;////////////need to new up these ...
        activation = a;
    }
}

void Layer::AddNodes(vector<int> = {}) {
    for (int i = 0; i < width; i++) {
        //Node node({});
        Node* node = new Node({});
        nodes.push_back(node);
    }
}

vector<Node*> Layer::GetInpNodes(vector<int> inputInds) {//// bad change this
    vector<Node*> _nodes;
    Node* tempN;
    if (prevLayer != NULL) {
        if (std::find(inputInds.begin(), inputInds.end(), prevLayer->index) != inputInds.end()) {
            //for (Node _node : prevLayer->nodes) {
            for(int i = 0; i < prevLayer->nodes.size(); i++){
                tempN = prevLayer->nodes[i];
                _nodes.push_back(tempN);
            }
        }
        if (prevLayer->prevLayer != nullptr) {
            vector<Node*> temp = prevLayer->prevLayer->GetInpNodes(inputInds);
            _nodes.insert(_nodes.end(), temp.begin(), temp.end());
        }
    }
    return _nodes;
}

void Layer::AddLayer(Layer* newLayer){
    if (nextLayer == NULL) {
        nextLayer = newLayer;
        (*newLayer).prevLayer = this;
    }
    else { (*nextLayer).AddLayer(newLayer); }

}

void Layer::EndBackProp() {
    for (int i = 0; i < nodes.size(); i++) {
        nodes[i]->desiredVals.clear();
    }
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
        //Node node(inpNodes);need to new up these nodes too    
        Node* nodeP = new Node(inpNodes);
        nodes.push_back(nodeP);
        i++;
    }
}

void Dense::StartBackProp(vector<float> desiredOut) {
    float derivActiv;
    for (int i = 0; i < nodes.size(); i++) {
        (nodes[i])->desiredVals.clear();
        (nodes[i])->desiredVals.push_back(desiredOut[i]);
        derivActiv = activation->DerivActivation(((nodes[i])->activation));
        (nodes[i])->SetPassChanges(derivActiv);
    }
    if (prevLayer->prevLayer != NULL) {
        prevLayer->BackProp();
    }
}

void Dense::BackProp() {
    float derivActiv;
    for (int i = 0; i < nodes.size(); i++) {
        derivActiv = activation->DerivActivation((nodes[i])->activation);
        (nodes[i])->SetPassChanges(derivActiv);
    }
    if (prevLayer->prevLayer != NULL) {
        prevLayer->BackProp();
    }
    else {
        prevLayer->EndBackProp();
    }
}

float Dense::GetCost(vector<float> desiredOut) {
    float cost = 0;
    for (int i = 0; i < nodes.size(); i++) {
        float temp;
        temp = (nodes[i])->activation - desiredOut[i];
        temp = temp * temp;
        cost += temp;
    }
    return cost;
}

void Dense::SetChanges(int batchSize) {
    for (Node* node : nodes) {
        node->AdjustWB(batchSize);
    }
    if (prevLayer->prevLayer != NULL) {
        prevLayer -> SetChanges(batchSize);
    }
}

Input::Input(int width) : Layer(width, "") {}

void Input::StartForwardProp(vector<float> input) {
    SetNodes(input);
    (*nextLayer).ForwardProp();
}

void Input::SetNodes(vector<float> input) {
    for (int i = 0; i < nodes.size(); i++) {
        (nodes[i])->activation = input[i];
    }
}