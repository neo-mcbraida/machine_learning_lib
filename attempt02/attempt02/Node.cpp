#pragma once
#include <iostream>
#include <vector>
#include "node.h"
#include <math.h>

using namespace std;
using namespace ntwrk;

Node::Node(vector<Node*> _inpNodes) {
    inpNodes = _inpNodes;
    float activation = 0, rawVal = 0;
    float bias = RandomVal();
    for (int i = 0; i < _inpNodes.size(); i++) {
        float weight = RandomVal();
        weights.push_back(weight);
        sumWBChanges.push_back(0);
    }
}

void Node::SetActivation() {
    float _rawVal = bias;
    float prevNode;
    for (int i = 0; i < weights.size(); i++) {
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
    bias += biasChange / batchSize;
    if (bias > 2) {
        bias = 2;
    }
    else if (bias < -2) {
        bias = -2;
    }
    biasChange = 0;
}

void Node::SetPassChanges(float derivActivation, float avDerCost) {
    //float avDerCost = GetAveDerCost();
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
    float weight = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / 2));//1- 2
    weight -= 1;
    return weight;
}