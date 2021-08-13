#pragma once
#include <iostream>
#include <vector>
#include "node.h"
#include <math.h>
#include <numeric>

using namespace std;
using namespace ntwrk;

Node::Node(vector<Node*> _inpNodes) {
    inpNodes = _inpNodes;
    float activation = 0, error = 0, EwrtO = 0, rawVal = 0, cost = 0;
    float deltaWeights = 0;
    float bias = RandomVal();
    //int ind = weightsPointer.size();
    //vector<float> Ws = *(new vector<float>());
    //weights = &weightsPointer;
    for (int i = 0; i < _inpNodes.size(); i++) {
        float weight = RandomVal();
        weights.push_back(weight);
        //(*weightsPointer)[index].push_back(weight);

        //vector<float>* weights = &(*weightsPointer)[index];
        sumWBChanges.push_back(0);
    }
    //(*weightsPointer).push_back(Ws);
    //weights = &Ws;
    //weights = &((*weightsPointer)[weightsPointer->size() -1]);
}

void Node::SetActivation() {
    float _rawVal = bias;
    float prevNode;
    for (int i = 0; i < inpNodes.size(); i++) {
        prevNode = (*(inpNodes[i])).activation;    
        float temp = weights[i] * prevNode;
        _rawVal += temp;
    }
    rawVal = _rawVal;
}

void Node::AdjustWB(int batchSize) {
    float change;
    for (int i = 0; i < sumWBChanges.size(); i++) {
        change = sumWBChanges[i] / batchSize;
        weights[i] += change;
        if (weights[i] > 2) {
            weights[i] = 2;
        }
        else if (weights[i] < -2) {
            weights[i] = -2;
        }
        sumWBChanges[i] = 0;
    }
}

float Node::RandomVal() {
    float weight = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / 2));//1- 2
    weight -= 1;
    return weight;
}

MemoryNode::MemoryNode(vector<Node*> inps) : Node(inps) {
    float prevVal = 0;
    float w = RandomVal();
    weights.push_back(w);
}//parent constructor is run first

void MemoryNode::SetActivation() {
    Node::SetActivation();
    rawVal += weights[weights.size()] * prevVal;
}