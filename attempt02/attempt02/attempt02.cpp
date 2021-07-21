// NNattempt02.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

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
            //rawInputNodes.push_back(_rawInputNodes[i]);
        }
        std::cout << this << std::endl;
    }
    void SetNode() {
        float _rawVal = 0;
        for (int i = 0; i < weights.size(); i++) {
            float prevNode = *(rawInputNodes[i]);
            float temp = weights[i] * prevNode;
            _rawVal += temp;
        }
        rawVal = _rawVal;
        ///
        std::cout << rawVal << std::endl;
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
    virtual void AddLayer(Layer* layer) {
        if (nextLayer == NULL) {
            SetNextLayer(layer);
            (*layer).SetPreviousLayer(this);////////////////////////////
        }
        else {
            NextLayer(layer);
        }
    }
    virtual void AddNodes() {
        vector<float*> nodeptrs = GetNodePtrs();
        for (int i = 0; i < width; i++) {
            Node node(nodeptrs);
            nodes.push_back(node);
        }
    }
    void SetNextLayer(Layer* layer) {
        nextLayer = layer;
    }
    void NextLayer(Layer* adress) {

    }
    void SetPreviousLayer(Layer* ptr) {
        prevLayer = ptr;
    }
    virtual void SetNodes() {
        for (int i = 0; i < nodes.size(); i++) {
            nodes[i].SetNode();
        }
    }
    void NextLayer() {
        //Layer l = *prevLayer;///////////////error reading prev ptr
        //std::cout << l.nodes[0].rawVal << std::endl;
        if (nextLayer != NULL) {
            Layer l = *nextLayer;
            (*nextLayer).SetNodes();
        }
    }
private:
    vector<float*> GetNodePtrs() {
        vector<float*> nodePtrs;
        int _layerInd = layerIndex;//layers start from one
        Layer* _prevLayerPtr = prevLayer;
        Layer _prevLayer = *prevLayer;
        //Layer _prevLayer = *_prevLayerPtr;
        while (_layerInd > 0) {
            if (std::find(inpLayers.begin(), inpLayers.end(), _layerInd) != inpLayers.end()) {
                vector<float*> shortPtrs = _prevLayer.GetLayerNodePtr();
                nodePtrs.insert(nodePtrs.end(), shortPtrs.begin(), shortPtrs.end());
            }
            _layerInd--;
            _prevLayer = *_prevLayerPtr;
        }
        std::reverse(nodePtrs.begin(), nodePtrs.end());
        return nodePtrs;
    }
    vector<float*> GetLayerNodePtr() {
        vector<float*> nodePtrs;
        for (int i = 0; i < nodes.size(); i++) {
            nodePtrs.push_back(&(nodes[i].rawVal));
        }
        return nodePtrs;
    }
};

class HiddenLayer : public Layer {
public:
    //vector<int> inputLayers;
    HiddenLayer(int _width, vector<int> _inputLayers) : Layer(_width, _inputLayers) {
        //inputLayers = _inputLayers;
    }
private:
};

class Dense : public HiddenLayer {
public:
    //Layer* prevLayer;
    Dense(int _width, vector<int> _inputLayers) : HiddenLayer(_width, _inputLayers) {

    }

private:
};

class Input : public Layer {
public:
    Input(int _width = 0) : Layer(_width) {
        AddNodes();
    }
    void AddLayer(Layer* layer) {
        if (nextLayer == NULL) {
            nextLayer = layer;//create addpreviousptr method
            (*layer).SetPreviousLayer(this);
        }
        else {
            (*nextLayer).AddLayer(layer);
        }
    }
    void AddNodes() override {
        for (int i = 0; i < width; i++) {
            Node n({});
            nodes.push_back(n);
        }
    }
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
    Input inputLayer;
    Network() {
    }
    void Input(Input layer) {
        IncrementDepth();
        inputLayer = layer;
        inputLayer.layerIndex = depth;
    }
    void HiddenLayer(Layer* layer) {
        IncrementDepth();
        inputLayer.AddLayer(layer);
        Layer temp = *layer;
        (*layer).layerIndex = depth;
        (*layer).AddNodes();
    }
    void Estimate(vector<float> inputData) {
        inputLayer.SetNodes(inputData);
    }
private:
    int depth = 0;
    void IncrementDepth() {
        depth++;
    }
};

int main()
{/////////////////PROBLEM, node pointers are pointing to the wrong address, not sure why
    //std::cout << "Hello World!\n";

    Network myNetwork;
    //Input inp(10);
    myNetwork.Input(Input(1));
    Dense layer1(1, { 1 });
    myNetwork.HiddenLayer(&layer1);//input layers start from 1
    //myNetwork.HiddenLayer();
    Dense layer2(1, { 2 });
    myNetwork.HiddenLayer(&layer2);
    vector<float> inputData = { 1 };
    myNetwork.Estimate(inputData);
    std::cout << 0 << std::endl;
}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
