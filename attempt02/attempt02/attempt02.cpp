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
            std::cout << "rawInputNodes addresses: " << rawInputNodes[0] << std::endl;////delete this
        }
    }
    void SetNode() {
        float _rawVal = 0;
        for (int i = 0; i < weights.size(); i++) {
            std::cout << "node address to be used: " << rawInputNodes[i] << std::endl;
            float prevNode = *(rawInputNodes[i]);
            float temp = weights[i] * prevNode;
            _rawVal += temp;
        }
        rawVal = _rawVal;
        ///
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
            std::cout << nodeptrs[0] << std::endl;
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
        NextLayer();
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
    vector<float*> GetNodePtrs() {//somewhere in here there is an error
        vector<float*> nodePtrs;
        int _prevlayerInd = layerIndex - 1;//layers start from one
        Layer* _prevLayerPtr = prevLayer;
        Layer _prevLayer = *prevLayer;
        //Layer _prevLayer = *_prevLayerPtr;
        while (_prevlayerInd > 0) {
            if (std::find(inpLayers.begin(), inpLayers.end(), _prevlayerInd) != inpLayers.end()) {
               // std::cout << "address from temp" << & (_prevLayer.nodes[0].rawVal) << std::endl;
               std::cout << "address from pointer to layer: " << &((*_prevLayerPtr).nodes[0].rawVal) << std::endl;
                
                vector<float*> shortPtrs = (*_prevLayerPtr).GetLayerNodePtr();
                //vector<float*> shortPtrs = _prevLayer.GetLayerNodePtr();
                nodePtrs.insert(nodePtrs.end(), shortPtrs.begin(), shortPtrs.end());
            }
            _prevlayerInd--;
            //_prevLayer = *_prevLayerPtr;
        }
        std::reverse(nodePtrs.begin(), nodePtrs.end());
        return nodePtrs;
    }
    vector<float*> GetLayerNodePtr() {
        vector<float*> nodePtrs;
        for (int i = 0; i < nodes.size(); i++) {
            float* tempPtr = &(nodes[i]).rawVal;
            nodePtrs.push_back(tempPtr);
            std::cout << "node pointers from GetLayerNodePtr: " << nodePtrs[0] << std::endl;
            //std::cout << &(nodePtrs[i]) << std::endl;//////delete this line
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
            //std::cout << "addNodes: " << & (n.rawVal) << std::endl;
            nodes.push_back(n);
        }
    }
    void SetNodes(vector<float> inputData) {
        for (int i = 0; i < inputData.size(); i++) {
            std::cout << "before setting input nodes: " << &(nodes[i].rawVal) << std::endl;
            nodes[i].rawVal = inputData[i];
            std::cout << "after setting input nodes:  " << &(nodes[i].rawVal) << std::endl;
        }
        NextLayer();
    }
private:
};

class Network {
public:    
    Input* inputLayer;//input layer stored by network was a copy rather than reference so pointers were not working.
    Network() {
    }
    void Input(Input* layer) {
        IncrementDepth();
        inputLayer = layer;

        //std::cout << inputLayer << std::endl;
        (*inputLayer).layerIndex = depth;
    }
    void HiddenLayer(Layer* layer) {
        IncrementDepth();
        (*inputLayer).AddLayer(layer);
        //std::cout << (*inputLayer).prevLayer << std::endl;
        Layer temp = *layer;//delete this line
        (*layer).layerIndex = depth;
        (*layer).AddNodes();
    }
    void Estimate(vector<float> inputData) {
        (*inputLayer).SetNodes(inputData);
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
    Input inp(1);
    //std::cout << &inp << std::endl;////////////////
    myNetwork.Input(&inp);
    std::cout << "actual input address: " << & (inp.nodes[0].rawVal) << std::endl;
    Dense layer1(1, { 1 });
    myNetwork.HiddenLayer(&layer1);//input layers start from 1
    //std::cout << &(layer1.nodes[0].rawVal) << std::endl;
    //myNetwork.HiddenLayer();
    Dense layer2(1, { 2 });
    myNetwork.HiddenLayer(&layer2);
    //std::cout << &(layer2.nodes[0].rawVal) << std::endl;
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
