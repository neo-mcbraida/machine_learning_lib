#include <iostream>
#include <vector>
#include "LSTMNode.h"
#include <math.h>

using namespace ntwrk;

LSTMNode::LSTMNode(vector<LSTMNode*> inps) {
    int no_of_inputs = inps.size();
    inpNodes = inps;
    state = 0;
    out = 0;
    for (int u = 0; u < 4; u++) {
        weights.push_back({});
        biases.push_back(RandomVal());
        Us.push_back(RandomVal());
        for (int i = 0; i < no_of_inputs; i++) {
            weights[u].push_back(RandomVal());
        }
    }
}//parent constructor is run first

void LSTMNode::SetActivation() {
    float temp_Fz = ReturnSum(weights[0], Us[0], biases[0]);
    float temp_Iz = ReturnSum(weights[1], Us[1], biases[1]);
    float temp_Az = ReturnSum(weights[2], Us[2], biases[2]);
    float temp_Oz = ReturnSum(weights[3], Us[3], biases[3]);
    Fz = sigmoid.Operate(temp_Fz);
    Iz = sigmoid.Operate(temp_Iz);
    Az = tanh(temp_Az);
    Oz = sigmoid.Operate(temp_Oz);
    SetState();
    SetOut();
}

float LSTMNode::ReturnSum(vector<float> weights, float U, float bias) {
    float sum = 0;
    for (int i = 0; i < weights.size(); i++) {
        sum += weights[i] * inpNodes[i]->out;
    }
    sum += U * out;
    sum += bias;
    return sum;
}

void LSTMNode::SetState() {
    state = (Az * Iz) + Fz * state;
}

void LSTMNode::SetOut() {
    out = tanh(state) * Oz;
}

float LSTMNode::RandomVal() {
    float weight = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / 2));//1- 2
    weight -= 1;
    return weight;
}