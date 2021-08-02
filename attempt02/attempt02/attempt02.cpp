// NNattempt02.cpp : This file contains the 'main' function. Program execution begins and ends there.
#include <iostream>
#include <vector>
#include <math.h>
#include <algorithm>
#include <string>
#include <fstream>
#include <ios>
#include <cstring>
#include <sstream>

/*using std::vector;
using std::string;
using std::pair;
using std::to_string;
using std::ios;
using std::fstream;
using std::ifstream;
using std::getline;
*/
using namespace std;

class ActivationFunction {//abstract class
public:
    string type = 0;
    virtual float Activation(float t) = 0;
    virtual float DerivActivation(float t) = 0;
    virtual void FinalActivation(vector<Node> nodes) {}
};

class Sigmoid : public ActivationFunction {
public:
    string type = "sigmoid";
    virtual float Activation(float weightsbias) override{
        float result = 1 / (1 + exp(-weightsbias));
        return result;
    }
    virtual float DerivActivation(float weightsbias) override{
        float sigmoidRes = Activation(weightsbias);
        float result = sigmoidRes * (1 - sigmoidRes);
        return result;
    }
};

class Relu : public ActivationFunction{
public:
    string type = "relu";
    virtual float Activation(float weightsBias) override{
        float rawVal = 0;
        if (weightsBias >= 0) {
            rawVal = weightsBias;
        }
        return rawVal;
    }
    virtual float DerivActivation(float weightsBias) override{
        if (weightsBias >= 0) { return 1; }
        else { return 0; }
    }
private:
};

class Node {
public:
    float setVal, rawVal, biasChanges = 0, bias;
    vector<Node*> rawInputNodes;
    vector<float> weights;
    vector<float> desiredVals, weightChanges, predictions, rawPredictions;
    ActivationFunction* activation;
    float sumDerivCost = 0;
    Node(vector<Node*> _rawInputNodes, ActivationFunction* _activation) {
        activation = _activation;
        rawInputNodes = _rawInputNodes; 
        float randomWeight;
        for (int i = 0; i < rawInputNodes.size(); i++) {
            randomWeight = RandomWeight();
            weights.push_back(randomWeight);
            weightChanges.push_back(0);
        }

    }
    void SetChanges(int batchSize) {
        for (int i = 0; i < weights.size(); i++) {
            weights[i] += weightChanges[i]/batchSize;
            weightChanges[i] = 0;
        }
        bias += biasChanges/batchSize;
        biasChanges = 0;
    }
    //sets the value of the node
    void SetBatchSize(int size) {
        for (int i = 0; i < size; i++) {
            predictions.push_back(0);
            rawPredictions.push_back(0);
        }
    }
    void SetNode() {
        SetRawVal();
        setVal = (*activation).Activation(rawVal);
    }
    void BackPropNode() {
        float derivactivation = (*activation).DerivActivation(rawVal);
        //float derivCost = 2 * (setVal - desiredVal);
        float derivCost = GetAverageCost();//for output layer desired vals will hold a single val, which is the intended output
        //sumDerivCost += derivCost;
        AdjustWeightsVals(derivactivation, derivCost);
        AdjustBias(derivactivation, derivCost);
    }
    void SetAverages() {////probs delete this
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
    float RandomWeight() {
        float weight = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / 2));
        weight -= 1;
        return weight;
    }
    float GetAverageCost() {
        float derivedCost = 0;
        float singleCost;
        for (int i = 0; i < desiredVals.size(); i++) {
            singleCost = 2 * (setVal - desiredVals[i]);
        }
        return derivedCost;
    }
    void AdjustWeightsVals(float derivActivation, float derivCost) {
        for (int i = 0; i < weights.size(); i++) {
            Node& node = *(rawInputNodes[i]);//node is a given input node from a previous layer
            WeightChange(i, node.rawVal, derivActivation, derivCost);
            SetPrevNodeGoal(node, weights[i], derivActivation, derivCost);
        }
    }
    void SetPrevNodeGoal(Node node, float weight, float derivActivation, float derivCost) {/////////////this may cause error dont know c++ well yet
        float nodeChange = weight * derivActivation * derivCost;
        node.desiredVals.push_back(nodeChange);
    }
    void WeightChange(int weightInd, float prevNode, float derivActivation, float derivCost) {
        //float derivCost = 2 * (setVal - desiredVal);
        float change = prevNode * derivCost * derivActivation;
        change = -1 * change;
        weightChanges[weightInd] += change;
    }
    void AdjustBias(float derivActivation, float derivCost) {
        float change = derivActivation * derivCost;
        change *= -1;
        biasChanges += change;
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

class SoftMax : public ActivationFunction {
public:
    string type = "softmax";
    virtual float Activation(float weightsBias) override {
        return exp(weightsBias);
    }
    void FinalActivation(vector<Node> nodes) {
        float sum = 0;
        for (Node node : nodes) {
            sum += node.rawVal;
        }
        for (Node node : nodes) {
            node.setVal = node.rawVal / sum;
        }
    }
    virtual float DerivActivation(float weightsBias) override {
        return Activation(weightsBias);//derivative of e^x = e^x(same as normal activation)
    }
};


class Layer {
public:
    vector<Node> nodes;
    vector<int> inpLayers;
    int width, layerIndex;
    Layer* nextLayer = NULL;
    Layer* prevLayer = NULL;
    ActivationFunction* activation;
    string activ;
    //layer stores instance of activation function, that can store any information required, such as softmax activation
    //activation can be done from layer class not node class because of this also.**read me tomorrow**
    Layer(int _width, string _activation, vector<int> _inpLayers = {}) {
        width = _width;
        inpLayers = _inpLayers;
        activ = _activation;
    }
    //adds layer ptr to chain of pointers
    virtual void AddLayer(Layer* layer) {
        if (nextLayer == NULL) {
            SetNextLayer(layer);
            (*layer).SetPreviousLayer(this);////////////////////////////
        }
        else {
            (*nextLayer).AddLayer(layer);
            //NextLayer();
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

    //invokes method to set values of all nodes in the next layer, unless
    //there is no next layer, (then i need to make it output values of node in
    //current layer
    //returns 1d array of pointers to all nodes required from a specific layer
    vector<Node*> GetLayerNodePtr() {
        vector<Node*> nodePtrs;
        for (int i = 0; i < nodes.size(); i++) {
            Node* tempPtr = &nodes[i];
            nodePtrs.push_back(tempPtr);
        }
        return nodePtrs;
    }
    void SetAverages() {////delete this probs
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
    void GetWeights(vector<vector<vector<float>>> weights) {
        int ind = weights.size();//index to add vector of layers weights
        for (int i = 0; i < weights[ind].size(); i++) {
            vector<float> nWeights = nodes[i].weights;
            nWeights.push_back(nodes[i].bias);
            weights[ind].push_back(nWeights);
        }
        if (nextLayer != NULL) {
            (*nextLayer).GetWeights(weights);
        }
    }
    void SetChanges(int batchSize) {
        for (Node node : nodes) {
            node.SetChanges(batchSize);
        }
        if (prevLayer != NULL) {
            (*prevLayer).SetChanges(batchSize);
        }
    }    
    void NextLayer() {
        if (nextLayer != NULL) {
            (*nextLayer).SetNodes();
        }
    }
    void SetNodes() {
        for (int i = 0; i < nodes.size(); i++) {
            nodes[i].SetNode();
        }
        if (activ != "") {
            (*activation).FinalActivation(nodes);
        }
        NextLayer();
    }
    //virtual void SetNodes() = 0;
private:
};

class HiddenLayer : public Layer {
public:
    HiddenLayer(int _width, vector<int> _inputLayers, string _activation) : Layer(_width, _activation, _inputLayers) {}
    //instantiates and adds n number of nodes

        //for each node in layer, set value of the node
    //then calls method to repeat this for next layer
    void AddNodes() {
        vector<Node*> nodeptrs = GetNodePtrs();
        for (int i = 0; i < width; i++) {
            //std::cout << nodeptrs[0] << std::endl;
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
            nodes[i].desiredVals.clear();
            nodes[i].desiredVals.push_back(desiredOuts[i]);
            nodes[i].BackPropNode();
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
        Layer _prevLayer = *prevLayer;//////////////////////error on this line 
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
    Input(int _width = 0) : Layer(_width, "") {
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
    SoftMax softmax;
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
    void Train(vector<vector<float>> inputData, vector<vector<float>> desiredOutputs, int epochs, int batchSize, bool shuffle=true) {//passes data to input layer, call next layers recursively
        for (int i = 0; i < epochs; i++) {
            std::cout << "epoch: " << (i+1) << "/" << epochs << std::endl;
            if (shuffle == true) {
                ShuffleData(inputData, desiredOutputs);
            }
            RunEpoch(inputData, desiredOutputs, epochs, batchSize);
        }
    }
    void Predict() {}
    void SaveModel(string fileName) {
        string fName = fileName + ".txt";
        Network& ref = *this;
        fstream file(fName, ios::out);

        // Writing the object's data in file
        file.write((char*)&ref, sizeof(ref));
        /*vector<vector<vector<float>>> weights;
        weights = GetWeights();
        std::fstream file;
        string fName = fileName + ".csv";
        file.open(fName, ios::out);
        for (int i = 0; i < weights.size(); i++) {
            for (int u = 0; u < weights[i].size(); u++) {
                for (float weight : weights[i][u]) {
                    file << to_string(weight) << ", ";
                }
                file << "; ";
            }
            file << "\n";
        }*/
    }
    void TestModel() {}
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
        else if ((*layer).activ == "softmax") {
            (*layer).activation = &softmax;
        }
    }
    void RunEpoch(vector<vector<float>> inputData, vector<vector<float>> desiredOutputs, int epochs, int batchSize) {
        (*inputLayer).SetBatchSize(batchSize);
        for (int i = 0; i < batchSize; i++) {
            batchDesiredOuts.push_back(vector<float> {});
        }
        //fill_n(batchDesiredOuts.begin(), batchSize, vector<float>{0});//////////////////////////uh oh, need to store batch cost by storing each cost for each node rather than each value for each node
        for (int i = 0; i < inputData.size(); i++) {
            ///batchDesiredOuts.erase(batchDesiredOuts.begin());
            ///batchDesiredOuts.push_back(desiredOutputs[i]);
            (*inputLayer).SetNodes(inputData[i]);
            BackPropegate(desiredOutputs[i]);
            if (i % batchSize == 0) {
                float cost = (*finalLayer).GetCost(desiredOutputs[i]);
                //vector<float> aveAim = GetAverageAim(desiredOutputs);
                SetChanges(batchSize);
                OutPutProgress(i, inputData.size(), cost);
            }
        }
    }
    void BackPropegate(vector<float> desiredOutput) {
        HiddenLayer & final = *finalLayer;
        //final.SetAverages();
        final.StartBackProp(desiredOutput);
    }
    void SetChanges(int batchSize) {
        HiddenLayer & final = *finalLayer;
        //final.SetAverages();
        final.SetChanges(batchSize);
    }
    void ShuffleData(vector<vector<float>> inputData, vector<vector<float>> desiredOutputs) {
        int seed = unsigned(std::time(0));
        std::srand(seed);
        std::random_shuffle(inputData.begin(), inputData.end());

        std::srand(seed);
        std::random_shuffle(desiredOutputs.begin(), desiredOutputs.end());
    }
    void OutPutProgress(int progress, int amount, float cost) {
        string output = to_string(progress) + "/" + to_string(amount) + " [";
        float am = (float)amount;
        float prog = (float)progress;
        int numEqs = (int)(round(((float)progress / (float)amount) * 25.0));
        for (int i = 0; i < 25; i++) {
            if (i <= numEqs) {
                output += "#";
            }
            else {
                output += "-";
            }
        }
        output += "] Cost: " + to_string(cost);
        std::cout << output << std::endl;
    }
    /*vector<vector<vector<float>>> GetWeights() {
        vector<vector<vector<float>>> weights;
        (*inputLayer).GetWeights(weights);
        return weights;
    }*/
};

void LoadModel(string fileName, Network network) {
    std::ifstream file;

    // Opening file in input mode
    file.open(fileName, ios::in);

    // Reading from file into object "obj"
    file.read((char*)&network, sizeof(network));
}

void GetData(string fileName, vector<vector<float>>* images, vector<vector<float>>* labels) {
    ifstream fin;
    fin.open(fileName);
    vector<vector<float>> all;
    vector<float> singleIm;
    vector<float> _labels;
    string line;
    string temp = "";
    int t, label;
    while (!fin.eof()) {
        label = 0;
        line = "";
        getline(fin, line);
        if (line == ""){
            break;
        }

        for (char& c : line) {
            if (c != ',') {
                temp += c;
            }
            else {
                float pixel = std::stof(temp);
                if (label == 0) {
                    label = 1;
                }
                else {
                    pixel /= 255;
                }
                singleIm.push_back(pixel);
                temp = "";
            }
        }
        t = singleIm[0];
        singleIm.erase(singleIm.begin());
        _labels.clear();
        for (int i = 0; i < 10; i++) {
            if (i == t) {
                _labels.push_back(1);
            }
            else {
                _labels.push_back(0);
            }
        }
        (*labels).push_back(_labels);
        (*images).push_back(singleIm);
        singleIm.clear();
    }
}

int main()
{
    Network myNetwork;
    Input inp(783);
    myNetwork.Input(&inp);
    Dense layer1(300, { 1 }, "relu");
    Dense layer2(100, { 2 }, "relu");
    Dense layer3(50, { 3 }, "relu");
    Dense layer4(25, { 4 }, "relu");
    myNetwork.AddHiddenLayer(&layer1);//input layers start from 1
    Dense layer5(10, { 5 }, "relu");
    myNetwork.AddHiddenLayer(&layer2);
    myNetwork.AddHiddenLayer(&layer3);
    myNetwork.AddHiddenLayer(&layer4);
    myNetwork.AddHiddenLayer(&layer5);
    //vector<float> inputData = { 1, 1 };

    vector<vector<float>> Trainimages;
    vector<vector<float>> Trainlabels;
    vector<vector<float>> Testimages;
    vector<vector<float>> Testlabels;
    GetData("C:/Users/nsmne/Documents/machine_learning_lib/fashion-mnist_test.csv", &Testimages, &Testlabels);
    GetData("C:/Users/nsmne/Documents/machine_learning_lib/fashion-mnist_train.csv", &Trainimages, &Trainlabels);
    //std::cout << to_string(Trainimages[0][0]);
    myNetwork.Train(Trainimages, Trainlabels, 2, 16);
};