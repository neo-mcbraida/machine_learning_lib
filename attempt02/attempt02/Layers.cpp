#pragma once
#include <iostream>
#include <vector>
#include "Layers.h"
#include "node.h"
#include "Activations.h"

using namespace std;
using namespace ntwrk;

Layer::Layer(int _width, string _activation) {
    totalError = 0;
	width = _width;
    activationName = _activation;
	SetActivation(_activation);
    nodes = {};
}

void Layer::BackProp() {}
void Layer::SetPrevEwrtR() {}
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
        Constant* a = new Constant;
        activation = a;
    }
}

void Layer::AddNodes() {
    for (int i = 0; i < width; i++) {
        //weightsPointer->push_back({});
        //vector<float>& temp = (*weightsPointer)[weightsPointer->size() - 1];
        Node* node = new Node({});
        nodes.push_back(node);
    }
}

vector<Node*> Layer::GetInpNodes(vector<int> inputInds) {
    vector<Node*> _nodes;
    Node* tempN;
    if (prevLayer != NULL) {
        if (std::find(inputInds.begin(), inputInds.end(), prevLayer->index) != inputInds.end()) {
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

void Layer::SaveLayer(string fName) {
    fName = fName + "/layer" + to_string(index);
    //const char* layer
}

void Layer::SumWeights(vector<float>& weights){
    for (Node* node : nodes) {
        for (float weight : node->weights) {
            weights.push_back(weight);
        }
    }
    if (nextLayer != NULL) {
        nextLayer->SumWeights(weights);
    }
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
        //weightsPointer->push_back({});
        //vector<float>& temp = (*weightsPointer)[weightsPointer->size() - 1];
        Node* nodeP = new Node(inpNodes);
        nodes.push_back(nodeP);
        i++;
    }
}

void Dense::StartBackProp(vector<float> desiredOut, Loss* lossFunc) {
    float deltaWeight;
    for (int i = 0; i < nodes.size(); i++) {
        Node& tempNode = *(nodes[i]);
        tempNode.EwrtX = lossFunc->GetDerLoss(tempNode.activation, desiredOut[i]); 
        deltaWeight = tempNode.EwrtX * tempNode.rawVal;
        for (float weightC : tempNode.sumWBChanges) {
            weightC -= deltaWeight;
        }
    }
    SetPrevEwrtA();
    prevLayer->BackProp();
}

void Dense::SetPrevEwrtA() {
    for (Node* node : nodes) {
        for (int i = 0; i < node->weights.size(); i++) {
            Node & n = *(node->inpNodes[i]);
            n.EwrtX += node->weights[i] * node->EwrtX;
        }
        node->EwrtX = 0;
    }
}

void Dense::BackProp() {
    for (Node* node : nodes) {
        node->EwrtX *= activation->DerivActivation(node->rawVal);
        for (int i = 0; i < node->weights.size(); i++) {
            node->sumWBChanges[i] -= node->inpNodes[i]->activation * node->EwrtX;
        }
    }
    if (prevLayer->prevLayer != NULL) {
        SetPrevEwrtA();
        prevLayer->BackProp();
    }
}

/*void Dense::BackProp() {
    float derivActiv;
    float derivCost;
    for (int i = 0; i < nodes.size(); i++) {
        Node& node = *nodes[i];
        derivActiv = activation->DerivActivation(node.rawVal);
        derivCost = lossFunc->GetDerLoss(node.activation, node.desiredVals);
        node.desiredVals.clear();
        node.SetPassChanges(derivActiv, derivCost);// need to be strict on loss and cost namings
    }
    if (prevLayer->prevLayer != NULL) {
        prevLayer->BackProp(lossFunc);
    }
    else {
        prevLayer->EndBackProp();
    }
}*/

float Dense::GetCost(vector<float> desiredOut) {
    float cost = 0;
    float temp;
    for (int i = 0; i < nodes.size(); i++) {
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