// NNattempt02.cpp : This file contains the 'main' function. Program execution begins and ends there.
#include <iostream>
#include <vector>
#include <math.h>

using std::vector;
using std::string;
//using std::exp;

class ActivationFunction {
public:
    virtual float Activation(float t) {
        return 0.0;
    }
    virtual float DerivActivation(float t) {
        return 0.0;
    }
};

class Sigmoid : public ActivationFunction {
public:
    float Activation(float weightsbias) override{
        float result = 1 / (1 + exp(-weightsbias));
        return result;
    }
    float DerivActivation(float weightsbias) override{
        float sigmoidRes = Activation(weightsbias);
        float result = sigmoidRes * (1 - sigmoidRes);
        return result;
    }
};

class Relu : public ActivationFunction{
public:
    float Activation(float weightsBias) override{
        float rawVal = 0;
        if (weightsBias >= 0) {
            rawVal = weightsBias;
        }
        return rawVal;
    }
    float DerivActivation(float weightsBias) override{
        if (weightsBias >= 0) { return 1; }
        else { return 0; }
    }
private:
};

class Node {
public:
    float setVal, rawVal, bias = 1;
    vector<Node*> rawInputNodes;
    vector<float> weights;
    vector<float> changes;
    vector<float> predictions;
    vector<float> rawPredictions;
    ActivationFunction* activation;
    Node(vector<Node*> _rawInputNodes, ActivationFunction* _activation) {
        activation = _activation;
        rawInputNodes = _rawInputNodes;
        for (int i = 0; i < rawInputNodes.size(); i++) {
            weights.push_back(0.5);//default to 0.5, (completely arbitrary number)
        }
    }
    //sets the value of the node
    void SetBatchSize(int size) {
        fill_n(predictions.begin(), size, 0);
        fill_n(rawPredictions.begin(), size, 0);
    }
    void SetNode() {
        SetRawVal();
        setVal = (*activation).Activation(rawVal);
        predictions.push_back(setVal);
        predictions.erase(predictions.begin());
        rawPredictions.push_back(rawVal);
        rawPredictions.erase(rawPredictions.begin());
        ///(below) for debuggin purposes
        std::cout << "raw value: " << rawVal << std::endl;
        std::cout << "raw value: " << setVal << std::endl;
        ///
    }
    void BackPropNode() {
        float desiredVal = GetDesiredVal();
        StartBackProp(desiredVal);
    }
    void StartBackProp(float desiredVal) {
        float derivactivation = (*activation).DerivActivation(rawVal);
        float derivCost = 2 * (setVal - desiredVal);
        AdjustWeightsVals(derivactivation, desiredVal, derivCost);
        AdjustBias(derivactivation, derivCost);
    }
    void SetAverages() {
        float _raw = 0, _set = 0;
        int bSize = predictions.size();
        for (int i = 0; i < bSize; i++) {
            _raw += rawPredictions[i];
            _set += predictions[i];
        }
        rawVal = _raw / bSize;
        setVal = _set / bSize;
    }
private:
    float GetDesiredVal() {
        float val = setVal;
        for (int i = 0; i < changes.size(); i++) {
            setVal += changes[i];
        }
        return val;
    }
    void AdjustWeightsVals(float derivActivation, float desiredVal, float derivCost) {
        for (int i = 0; i < weights.size(); i++) {
            Node& node = *(rawInputNodes[i]);
            WeightChange(weights[i], node.rawVal, derivActivation, desiredVal);
            NodeChange(node, weights[i], derivActivation, derivCost);
        }
    }
    void NodeChange(Node node, float weight, float derivActivation, float derivCost) {
        float nodeChange = weight * derivActivation * derivCost;
        node.changes.push_back(nodeChange);
    }
    void WeightChange(float weight, float prevNode, float derivActivation, float desiredVal) {
        float derivCost = 2 * (setVal - desiredVal);
        float change = prevNode * derivCost * derivActivation;
        change = -1 * change;
        weight += change;
    }
    void AdjustBias(float derivActivation, float derivCost) {
        float change = derivActivation * derivCost;
        change *= -1;
        bias += change;
    }
    void SetRawVal() {
        float _rawVal = bias;
        for (int i = 0; i < weights.size(); i++) {
            float prevNode = (*(rawInputNodes[i])).rawVal;
            float temp = weights[i] * prevNode;
            _rawVal += temp;
        }
        rawVal = _rawVal;
    }
};

class Layer {
public:
    vector<Node> nodes;
    vector<int> inpLayers;
    int width, layerIndex;
    Layer* nextLayer = NULL;
    Layer* prevLayer = NULL;
    Layer(int _width, vector<int> _inpLayers = {}) {
        width = _width;
        inpLayers = _inpLayers;
    }

    //adds layer ptr to chain of pointers
    virtual void AddLayer(Layer* layer) {
        if (nextLayer == NULL) {
            SetNextLayer(layer);
            (*layer).SetPreviousLayer(this);////////////////////////////
        }
        else {
            NextLayer();
        }
    }
    //sets pointer to the next layer
    void SetNextLayer(Layer* layer) {
        nextLayer = layer;
    }

    //sets pointer to the previous layer
    void SetPreviousLayer(Layer* ptr) {
        prevLayer = ptr;
    }
    void SetBatchSize(int size) {
        for (int i = 0; i < nodes.size(); i++) {
            nodes[i].SetBatchSize(size);
        }
        if (nextLayer != NULL) {
            (*nextLayer).SetBatchSize(size);
        }
    }
    //for each node in layer, set value of the node
    //the calls method to repeat this for next layer
    virtual void SetNodes() {
        for (int i = 0; i < nodes.size(); i++) {
            nodes[i].SetNode();
        }
        NextLayer();
    }

    //invokes method to set values of all nodes in the next layer, unless
    //there is no next layer, (then i need to make it output values of node in
    //current layer
    void NextLayer() {
        if (nextLayer != NULL) {
            (*nextLayer).SetNodes();
        }
    }    
    //returns 1d array of pointers to all nodes required from a specific layer
    vector<Node*> GetLayerNodePtr() {
        vector<Node*> nodePtrs;
        for (int i = 0; i < nodes.size(); i++) {
            Node* tempPtr = &nodes[i];
            nodePtrs.push_back(tempPtr);
        }
        return nodePtrs;
    }
    void SetAverages() {
        for (int i = 0; i < nodes.size(); i++) {
            nodes[i].SetAverages();
        }
        if (prevLayer != NULL) {
            (*prevLayer).SetAverages();
        }
    }    
    void BackProp() {
        for (Node node : nodes) {
            node.BackPropNode();
        }
        if ((*prevLayer).prevLayer != NULL) {
            (*prevLayer).BackProp();
        }
    }
private:
};

class HiddenLayer : public Layer {
public:
    ActivationFunction* activation;
    string activ;
    HiddenLayer(int _width, vector<int> _inputLayers, string _activation) : Layer(_width, _inputLayers) {
        activ = _activation;
    }
    //instantiates and adds n number of nodes
    void AddNodes() {
        vector<Node*> nodeptrs = GetNodePtrs();
        for (int i = 0; i < width; i++) {
            std::cout << nodeptrs[0] << std::endl;
            Node node(nodeptrs, activation);
            nodes.push_back(node);
        }
    }
    vector<float> GetDifferences(vector<float> desiredVals) {
        vector<float> diffs;
        for (int i = 0; i < desiredVals.size(); i++) {
            float dif;
            dif = desiredVals[i] - nodes[i].rawVal;
            diffs.push_back(dif);
        }
        return diffs;
    }
    float GetCost(vector<float> desiredVals) {
        vector<float> diffs = GetDifferences(desiredVals);
        float cost = 0;
        for (int i = 0; i < diffs.size(); i++) {
            float temp = diffs[i];
            cost += temp * temp;
        }
        return cost;
    }
    void StartBackProp(vector<float> desiredOuts) {
        for (int i = 0; i < desiredOuts.size(); i++) {
            nodes[i].StartBackProp(desiredOuts[i]);
        }
        if((*prevLayer).prevLayer != NULL) {
            (*prevLayer).BackProp();
        }
    }
private:
    //Returns a 1d array of all pointers to all nodes required for the current layer
    vector<Node*> GetNodePtrs() {
        vector<Node*> nodePtrs;
        int _prevlayerInd = layerIndex - 1;//layers start from one
        Layer* _prevLayerPtr = prevLayer;
        Layer _prevLayer = *prevLayer;
        while (_prevlayerInd > 0) {
            if (std::find(inpLayers.begin(), inpLayers.end(), _prevlayerInd) != inpLayers.end()) {
                vector<Node*> shortPtrs = (*_prevLayerPtr).GetLayerNodePtr();
                nodePtrs.insert(nodePtrs.end(), shortPtrs.begin(), shortPtrs.end());
            }
            _prevlayerInd--;
        }
        std::reverse(nodePtrs.begin(), nodePtrs.end());
        return nodePtrs;
    }
};

class Dense : public HiddenLayer {
public:
    Dense(int _width, vector<int> _inputLayers, string _activation) : HiddenLayer(_width, _inputLayers, _activation) {

    }
private:
};

class Input : public Layer {
public:
    ActivationFunction* activation;
    Input(int _width = 0) : Layer(_width) {
        AddNodes();
    }

    //adds layer pointer to the chain of pointers
    //takes new layer pointer as argument
    void AddLayer(Layer* layer) {
        if (nextLayer == NULL) {
            nextLayer = layer;
            (*layer).SetPreviousLayer(this);
        }
        else {
            (*nextLayer).AddLayer(layer);
        }
    }

    //adds nodes to the layer, depending on it's width
    void AddNodes(){
        for (int i = 0; i < width; i++) {
            Node n({}, activation);
            nodes.push_back(n);
        }
    }

    //for every node in layer, calculates value of that node
    //then calls function to repeate this for next layer
    void SetNodes(vector<float> inputData) {
        for (int i = 0; i < inputData.size(); i++) {
            nodes[i].rawVal = inputData[i];
        }
        NextLayer();
    }
private:
};

class Network {
public:
    Sigmoid sigmoid;
    Relu relu;
    Input* inputLayer;
    HiddenLayer* finalLayer;
    vector<vector<float>> batchDesiredOuts;
    Network() {
    }
    void Input(Input* layer) {//adds input layer to network, input layer must be first layer
        IncrementDepth();
        inputLayer = layer;
        (*inputLayer).layerIndex = depth;
    }
    void AddHiddenLayer(HiddenLayer* layer) {//adds hidden layer to network, takes ptr as argument
        IncrementDepth();
        SetLayerActivation(layer);
        (*inputLayer).AddLayer(layer);
        (*layer).layerIndex = depth;
        (*layer).AddNodes();
        finalLayer = layer;
    }
    void Estimate(vector<vector<float>> inputData, vector<vector<float>> desiredOutputs, int epochs, int batchSize) {//passes data to input layer, call next layers recursively
        (*inputLayer).SetBatchSize(batchSize);
        fill_n(batchDesiredOuts.begin(), batchSize, vector<float>{0});
        for (int i = 0; i < inputData.size(); i++) {
            batchDesiredOuts.erase(batchDesiredOuts.begin());
            batchDesiredOuts.push_back(desiredOutputs[i]);
            (*inputLayer).SetNodes(inputData[i]);
            if (i % batchSize == 0) {
                vector<float> aveAim = GetAverageAim(desiredOutputs);
                BackPropegate(aveAim);
            }
        }
    }
    void BackPropegate(vector<float> desiredOutput){
        HiddenLayer& final = *finalLayer;
        final.SetAverages();
        final.StartBackProp(desiredOutput);
    }
private:
    int depth = 0;//number of layers in network
    void IncrementDepth() {//does what it says on the tin
        depth++;
    }
    vector<float> GetAverageAim(vector<vector<float>> outputs) {
        vector<float> average;
        for (int i = 0; i < outputs[0].size(); i++) {
            float temp = 0;
            for (int u = 0; u < outputs.size(); u++) {
                temp += outputs[u][i];
            }
            temp /= outputs.size();
            average.push_back(temp);
        } 
        return average;
    }
   /* float CostFunction(vector<float> actualVals, vector<float> output) {
        int cost = 0;
        for (int i = 0; i < actualVals.size(); i++) {
            int temp = actualVals[i] * output[i];
            cost += temp * temp;
        }
        return cost;
    } */
    void SetLayerActivation(HiddenLayer* layer) {
        if ((*layer).activ == "relu") {
            (*layer).activation = &relu;
        }
        else if ((*layer).activ == "sigmoid") {
            (*layer).activation = &sigmoid;
        }
    }
};

int main()
{
    Network myNetwork;
    Input inp(2);
    myNetwork.Input(&inp);
    Dense layer1(2, { 1 }, "relu");
    myNetwork.AddHiddenLayer(&layer1);//input layers start from 1
    Dense layer2(2, { 1, 2 }, "sigmoid");
    myNetwork.AddHiddenLayer(&layer2);
    vector<float> inputData = { 1, 1 };
    //myNetwork.Estimate(inputData, 2, 16);
};