#include <iostream>
#include <vector>
#include "LSTMNode.h"
#include <math.h>

using namespace ntwrk;

LSTMNode::LSTMNode(vector<LSTMNode*> inps) {
    int no_of_inputs = inps.size();
    inpNodes = inps;
    state.push_back(0);
    out.push_back(0);
    for (int u = 0; u < 4; u++) {
        weights.push_back({});
        weightChange.push_back({});
        biases.push_back(RandomVal());
        //Us.push_back(RandomVal());
        weights[u].push_back(RandomVal());
        for (int i = 0; i < no_of_inputs; i++) {
            weights[u].push_back(RandomVal());
        }
    }
    //Sigmoid sigmoid;
    //Tanh tanH;
}//parent constructor is run first

void LSTMNode::SetActivation() {
    vector<float> timeStepInps = {};
    for (LSTMNode* node : inpNodes) {
        timeStepInps.push_back(out[out.size() - 1]);
        timeStepInps.push_back(node->out[out.size() - 1]);
    }
    inputs.push_back(timeStepInps);
    float temp_Fz = ReturnSum(weights[0], biases[0]);
    float temp_Iz = ReturnSum(weights[1], biases[1]);
    float temp_Az = ReturnSum(weights[2], biases[2]);
    float temp_Oz = ReturnSum(weights[3], biases[3]);
    temp_Fz = sigmoid.Operate(temp_Fz);
    Fz.push_back(temp_Iz);
    temp_Iz = sigmoid.Operate(temp_Iz);
    Iz.push_back(temp_Iz);
    temp_Az = tanh(temp_Az);
    Az.push_back(temp_Iz);
    temp_Oz = sigmoid.Operate(temp_Oz);
    Oz.push_back(temp_Iz);
    SetState();
    SetOut();
}

float LSTMNode::ReturnSum(vector<float> weights, float bias) {
    float sum = 0;
    for (int i = 0; i < weights.size(); i++) {
        sum += weights[i] * inputs[inputs.size() - 1][i];
    }
    sum += bias;
    return sum;
}

void LSTMNode::SetState() {
    float tAz = Az[Az.size() - 1];
    float tIz = Iz[Iz.size() - 1];
    float tFz = Fz[Fz.size() - 1];
    float tState = state[state.size() - 1];
    state.push_back((tAz * tIz) + (tFz * tState));
}

void LSTMNode::SetOut() {
    float temp = state[state.size()];
    out.push_back(tanh(temp) * Oz[Oz.size() - 1]);
}

float LSTMNode::RandomVal() {
    float weight = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / 2));//1- 2
    weight -= 1;
    return weight;
}

void LSTMNode::BackProp(float nextState, float nextForget, float NextErrorbyActiv) {//remove NextErrorByActiv
    int timeSteps = state.size();
    // back prop through time starting at t = t
    float nextState = 0;
    float nextForget = 0;
    for (int t = timeSteps; t > 0; t--) {// at this point total loss should already have a value
        float common_deriv = (totalErrors[t] * GetActivationByState(t)) + nextForget;
        for (int i = 0; i < inpNodes.size(); i++) {
            AddWeightChange(t, nextState, common_deriv);
        }
        SetPrevError(common_deriv, t);
        nextState = state[t];
        nextForget = Fz[t];
    }
}

void LSTMNode::AddWeightChange(int time, float nextState, float common_deriv) {
    for (int i = 0; i < weights[0].size(); i++) {
        float deltaFz = common_deriv * GetCByWf(sums[0][time], inputs[0][time], time);
        float deltaIz = common_deriv * GetCByWf(sums[1][time], inputs[1][time], time);
        float deltaAz = common_deriv * GetCByWf(sums[2][time], inputs[2][time], time);
        float deltaWz = common_deriv * GetAByWo(sums[3][time], inputs[3][time], time);
        weightChange[0].push_back(deltaFz);
        weightChange[1].push_back(deltaIz);
        weightChange[2].push_back(deltaAz);
        weightChange[3].push_back(deltaWz);
        float prevNodeError = 0;
        prevNodeError += weights[0][i] * totalErrors[time];
        prevNodeError += weights[1][i] * totalErrors[time];
        prevNodeError += weights[2][i] * totalErrors[time];
        prevNodeError += weights[3][i] * totalErrors[time];
        inpNodes[i]->totalErrors.insert(totalErrors.begin(), prevNodeError);
    }
}

float LSTMNode::GetStateByPrevState(int index) { return Fz[index]; }// should probably remove this

float LSTMNode::GetActivationByState(int index) {
    float temp = tanh(state[index]);
    float result = Oz[index] * (1 - (temp * temp));
    return result;
}

float LSTMNode::GetNextStateByState(int index) {}// can remove this

float LSTMNode::GetAByWo(float sum, float input, int index) {
    float tempState = state[index];
    float result = tempState * sigmoid.DerivActivation(sum) * input;
    return result;
}

float LSTMNode::GetCByWf(float sum, float input, int index) {
    float result = state[index - 1] * sigmoid.DerivActivation(sum) * input;
    return result;
}

float LSTMNode::GetCByWu(float sum, float input, int index) {
    float tempAz = Az[index];
    float result = tempAz * sigmoid.DerivActivation(sum) * input;
    return result;
}

float LSTMNode::GetCByWc(float sum, float input, int index) {
    //Iz
    float tempIz = Iz[index];
    float temp = tanh(sum);
    float result = tempIz * (1 - (temp * temp)) * input;
    return result;
}

float LSTMNode::SetPrevError(float common_deriv, int index) {
    float temp = state[index - 1] * (Fz[index] * (1 - Fz[index])) * out[index - 1];
    float result = 3 * temp * common_deriv;
    temp = tanh(state[index]) * sigmoid.DerivActivation(sums[3][index]) * weights[3][0];
    result += totalErrors[index] * temp;
    totalErrors[index - 1] += (result);
}

// need to add attribute to store current timestep during backprop