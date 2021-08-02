#pragma once
#include <iostream>
#include <vector>
#include "node.h"

using namespace std;
using namespace ntwrk;

Node::Node(vector<Node*> _inpNodes) {
    inpNodes = _inpNodes;
}

void Node::SetActivation() {
    float _rawVal = bias;
    for (int i = 0; i < weights.size(); i++) {
        float prevNode = (*(inpNodes[i])).activation;
        float temp = weights[i] * prevNode;
        _rawVal += temp;
    }
    activation = _rawVal;
}

void Node::AdjustWB(int batchSize) {
    float change;
    for (int i = 0; i < sumWBChanges.size(); i++) {
        change = sumWBChanges[i] / batchSize;
        weights[i] += change;
        sumWBChanges[i] = 0;
    }
    bias += biasChange / batchSize;
    biasChange = 0;
}

void Node::SetPassChanges(float derivActivation) {
    float avDerCost = GetAveDerCost();
    for (int i = 0; i < inpNodes.size(); i++) {
        SetPrevNodeDesire(avDerCost, i, derivActivation);
        SetWeightGrad(avDerCost, i, derivActivation);
    }
    SetBiasGrad(avDerCost, derivActivation);
}

float Node::GetAveDerCost() {
    float total = 0;
    int amount = desiredVals.size();
    for (int i = 0; i < amount; i++) {
        total += (2 * (activation - desiredVals[i]));
    }
    total /= amount;
    return total;
}

void Node::SetPrevNodeDesire(float avDerCost, int nodeInd, float derivActivation) {
    float desVal = weights[nodeInd] * avDerCost * derivActivation;
    (*inpNodes[nodeInd]).desiredVals.push_back(desVal);
}

void Node::SetWeightGrad(float avDerCost, int nodeInd, float derivActivation) {
    float change;
    change = (*inpNodes[nodeInd]).activation * avDerCost * derivActivation;
    sumWBChanges[nodeInd] += change;
}

void Node::SetBiasGrad(float avDerCost, float derivActivation) {
    float change = derivActivation * avDerCost;
    change *= -1;
    biasChange += change;
}

float Node::RandomVal() {
    float weight = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / 2));
    weight -= 1;
    return weight;
}