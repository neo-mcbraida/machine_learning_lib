// NNattempt02.cpp : This file contains the 'main' function. Program execution begins and ends there.
#include <iostream>
#include <vector>

using std::vector;

class Node {
public:
    float rawVal;
    vector<float*> rawInputNodes;
    vector<float> weights;
    Node(vector<float*> _rawInputNodes) {
        rawInputNodes = _rawInputNodes;
        for (int i = 0; i < rawInputNodes.size(); i++) {
            weights.push_back(0.5);//default to 0.5, (completely arbitrary number)
        }
    }

    //sets the value of the node
    void SetNode() {
        float _rawVal = 0;
        for (int i = 0; i < weights.size(); i++) {
            float prevNode = *(rawInputNodes[i]);
            float temp = weights[i] * prevNode;
            _rawVal += temp;
        }
        rawVal = _rawVal;
        ///(below) for debuggin purposes
        std::cout << "raw value: " << rawVal << std::endl;
        ///
    }
private:
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

    //instantiates and adds n number of nodes
    virtual void AddNodes() {
        vector<float*> nodeptrs = GetNodePtrs();
        for (int i = 0; i < width; i++) {
            std::cout << nodeptrs[0] << std::endl;
            Node node(nodeptrs);
            nodes.push_back(node);
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
private:
    //Returns a 1d array of all pointers to all nodes required for the current layer
    vector<float*> GetNodePtrs() {
        vector<float*> nodePtrs;
        int _prevlayerInd = layerIndex - 1;//layers start from one
        Layer* _prevLayerPtr = prevLayer;
        Layer _prevLayer = *prevLayer;
        while (_prevlayerInd > 0) {
            if (std::find(inpLayers.begin(), inpLayers.end(), _prevlayerInd) != inpLayers.end()) {
                vector<float*> shortPtrs = (*_prevLayerPtr).GetLayerNodePtr();
                nodePtrs.insert(nodePtrs.end(), shortPtrs.begin(), shortPtrs.end());
            }
            _prevlayerInd--;
        }
        std::reverse(nodePtrs.begin(), nodePtrs.end());
        return nodePtrs;
    }

    //returns 1d array of pointers to all nodes required from a specific layer
    vector<float*> GetLayerNodePtr() {
        vector<float*> nodePtrs;
        for (int i = 0; i < nodes.size(); i++) {
            float* tempPtr = &(nodes[i]).rawVal;
            nodePtrs.push_back(tempPtr);
        }
        return nodePtrs;
    }
};

class HiddenLayer : public Layer {
public:
    HiddenLayer(int _width, vector<int> _inputLayers) : Layer(_width, _inputLayers) {
    }
private:
};

class Dense : public HiddenLayer {
public:
    Dense(int _width, vector<int> _inputLayers) : HiddenLayer(_width, _inputLayers) {

    }
private:
};

class Input : public Layer {
public:
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
    void AddNodes() override {
        for (int i = 0; i < width; i++) {
            Node n({});
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
    Input* inputLayer;
    Network() {
    }
    void Input(Input* layer) {//adds input layer to network, input layer must be first layer
        IncrementDepth();
        inputLayer = layer;
        (*inputLayer).layerIndex = depth;
    }
    void HiddenLayer(Layer* layer) {//adds hidden layer to network, takes ptr as argument
        IncrementDepth();
        (*inputLayer).AddLayer(layer);
        (*layer).layerIndex = depth;
        (*layer).AddNodes();
    }
    void Estimate(vector<float> inputData) {//passes data to input layer, call next layers recursively
        (*inputLayer).SetNodes(inputData);
    }
private:
    int depth = 0;//number of layers in network
    void IncrementDepth() {//does what it says on the tin
        depth++;
    }
};

int main()
{
    Network myNetwork;
    Input inp(2);
    myNetwork.Input(&inp);
    Dense layer1(2, { 1 });
    myNetwork.HiddenLayer(&layer1);//input layers start from 1
    Dense layer2(2, { 1, 2 });
    myNetwork.HiddenLayer(&layer2);
    vector<float> inputData = { 1, 1 };
    myNetwork.Estimate(inputData);
}