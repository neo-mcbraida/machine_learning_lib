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
    totalError = 0;
	width = _width;
	SetActivation(_activation);
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

void Dense::StartBackProp(vector<float> desiredOut, Loss* lossFunc) {
    //float derivActiv;
    //float derivLoss;
    float deltaWeight;
    for (int i = 0; i < nodes.size(); i++) {
        Node& tempNode = *(nodes[i]);
        //tempNode.desiredVals.clear();
        //tempNode.desiredVals.push_back(desiredOut[i]);
        //derivActiv = activation->DerivActivation(tempNode.rawVal);
        tempNode.EwrtX = lossFunc->GetDerLoss(tempNode.activation, desiredOut[i]); 
        deltaWeight = tempNode.EwrtX * tempNode.rawVal;
        for (float weightC : tempNode.sumWBChanges) {
            weightC -= deltaWeight;
        }
        //totalError += tempNode.EwrtR;
        //tempNode.SetPassChanges(derivActiv, derivLoss);
    }
    /*for (Node* node : nodes) {
        float deltaWeights = node->EwrtR * node->rawVal;
        node->deltaWeights += deltaWeights;
    }*/
    SetPrevEwrtA();

    prevLayer->BackProp();
    //if (prevLayer->prevLayer != NULL) {
    //    prevLayer->BackProp(lossFunc);
    //}
}

void Dense::SetPrevEwrtA() {
    //float totalEwrtO = 0;
    for (Node* node : nodes) {
        for (int i = 0; i < node->weights.size(); i++) {
            Node & n = *(node->inpNodes[i]);
            n.EwrtX += node->weights[i] * node->EwrtX;
            //float temp = node->inpNodes[i]->EwrtX;
            //n.activation->DerivActivation(temp);
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